#
# train.py - Unified Latent Diffusion Training
# Automatically detects and loads the correct autoencoder (MLP or Conv) from checkpoint
#

from pathlib import Path
import sys

# Add parent directory to path for common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, argparse, time
import torch
import torch.nn.functional as F
from torchvision import utils as tvutils
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext

# Import common utilities
from common.device import get_device, seed_everything, seed_worker
from common.data import HF_MNIST, make_mnist_loader
from common.logger_utils import setup_component_logger, log_args_rich
from common.slurm import install_signal_handlers, stop_requested
from common import error_codes
from common import emoji
from common.ckpt import resolve_resume_path, load_checkpoint, save_checkpoint_and_link_latest
import logging

# Import BOTH AE model modules
import importlib.util

# Import MLP autoencoder
spec_mlp = importlib.util.spec_from_file_location("ae_module_mlp",
                                                    str(Path(__file__).parent / "../mnist_ae/model.py"))
ae_module_mlp = importlib.util.module_from_spec(spec_mlp)
spec_mlp.loader.exec_module(ae_module_mlp)

# Import Conv autoencoder
spec_conv = importlib.util.spec_from_file_location("ae_module_conv",
                                                     str(Path(__file__).parent / "../mnist_ae/model_conv.py"))
ae_module_conv = importlib.util.module_from_spec(spec_conv)
spec_conv.loader.exec_module(ae_module_conv)

# Latent diffusion model (local) - import with explicit path
spec_local = importlib.util.spec_from_file_location("local_model", str(Path(__file__).parent / "model.py"))
local_model = importlib.util.module_from_spec(spec_local)
spec_local.loader.exec_module(local_model)

LatentDenoiser = local_model.LatentDenoiser
precompute_schedules = local_model.precompute_schedules
q_sample = local_model.q_sample
generate_images = local_model.generate_images

def get_ae_information_from_checkpoint(resume_ckpt_path: Path):
    try:
        print(f"{emoji.info} Extracing AE information from checkpoint")
        ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt["args"], dict):
            ae_ckpt = ckpt["args"]["ae_ckpt"]
            ae_latent_dim = ckpt["args"]["ae_latent_dim"]
            ae_model_type = ckpt["args"]["ae_model"]
            ae_only_digit = ckpt["args"]["ae_only_digit"]
        else:
            ae_ckpt = ckpt["args"].ae_ckpt
            ae_latent_dim = ckpt["args"].ae_latent_dim
            ae_model_type = ckpt["args"].ae_model
            ae_only_digit = ckpt["args"].ae_only_digit

        ae_model_type2, ae_latent_dim2, ae_only_digit2, ae_ckpt_data = infer_latent_dim_from_checkpoint(ae_ckpt)
        if ae_model_type != ae_model_type2 or ae_latent_dim != ae_latent_dim2 or ae_only_digit != ae_only_digit2:
            print(f"{emoji_error} Information in AE checkpoint {ae_ckpt} does not match with information stored in checkpoint: {resume_ckpt_path}") 
            exit(error_codes.MODEL_MISMATCH)

        return ae_model_type, ae_latent_dim, ae_only_digit, ae_ckpt, ae_ckpt_data
    except:
        print(f"{emoji.error} Cannot extract AE information from checkpoint. Exiting")
        exit(error_codes.NO_AE_CHECKPOINT)

def infer_latent_dim_from_checkpoint(ckpt_path: Path):
    """Infer latent_dim and model_type from autoencoder checkpoint args"""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle both dict and argparse.Namespace
        if isinstance(ckpt["args"], dict):
            latent_dim = ckpt["args"]["latent_dim"]
            model_type = ckpt["args"]["model"]
            only_digit = ckpt["args"]["only_digit"]
        else:
            latent_dim = ckpt["args"].latent_dim
            model_type = ckpt["args"].model
            only_digit = ckpt["args"].only_digit

        return model_type, latent_dim, only_digit, ckpt
    except:
        print(f"{emoji.error} Cannot load AE checkpoint: {ckpt_path}.  Exiting")
        exit(error_codes.NO_AE_CHECKPOINT)

def load_autoencoder(model, latent_dim, only_digit, ckpt, device, train_logger):
    if model == "ae-conv":
        ae_module = ae_module_conv
    elif model == "ae-mlp":
        ae_module = ae_module_mlp
    else:
        raise ValueError(f"Unknown autoencoder model type: {model}")

    ae = ae_module.Autoencoder(latent_dim=latent_dim, use_sigmoid=True).to(device)
    ckpt_device = {}
    for k, v in ckpt["model"].items():
        if isinstance(v, torch.Tensor):
            ckpt_device[k] = v.to(device)
        else:
            ckpt_device[k] = v
    ae.load_state_dict(ckpt_device)
    ae.eval()
    return ae.encoder, ae.decoder

def main():
    parser = argparse.ArgumentParser("Unified Latent DDPM MNIST trainer")
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("SLURM_TMPDIR", "./hf_cache"),
                        help="HuggingFace cache directory")
    parser.add_argument("--outdir", type=str, default=os.environ.get("SCRATCH", "./outputs"))
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--ae-ckpt", type=str, default=None, help="Path to pretrained autoencoder checkpoint")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Early stopping patience (epochs)")
    parser.add_argument("--T", type=int, default=1000, help="Diffusion steps")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for denoiser")
    parser.add_argument("--samples-per-epoch", type=int, default=16, help="Images to generate per epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ckpt-every-sec", type=int, default=300)
    parser.add_argument("--ckpt-save-every", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--auto-resume", action="store_true", default=False)
    parser.add_argument("--generate-images", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    args.model = "lddpm"

    #
    # Setup log dir
    #
    outdir = Path(args.outdir)
    log_dir = outdir / Path("logs") if args.logdir is None else Path(args.logdir)
    log_dir.mkdir(parents=True, exist_ok=True)

    #
    # Resume logic - determine checkpoint path and experiment name
    #
    exp_name, resume_ckpt_path, _ = resolve_resume_path(args)
    if args.auto_resume and not exp_name:
        print(f"{emoji.warning} Ignoring auto-resume, training from scratch")

    if not exp_name:
        print(f"{emoji.info} Loading specified AE checkpoint: {args.ae_ckpt}") 
        args.ae_model, args.ae_latent_dim, args.ae_only_digit, ae_ckpt_data = infer_latent_dim_from_checkpoint(Path(args.ae_ckpt))
        print(f"{emoji.info} AE model: {args.ae_model}")
        print(f"{emoji.info} AE model latent dim: {args.ae_latent_dim}")
        print(f"{emoji.info} AE model only digit: {args.ae_only_digit}")
    else:
        args.ae_model, args.ae_latent_dim, args.ae_only_digit, ae_ckpt_from_resume_ckpt, ae_ckpt_data_from_resume_ckpt = get_ae_information_from_checkpoint(resume_ckpt_path)
        if args.ae_ckpt:
            print(f"{emoji.warning} AE checkpoint loaded from {resume_ckpt_path} is ignored: {ae_ckpt_from_resume_ckpt}")
            
            ae_model_type2, ae_latent_dim2, ae_only_digit2, ae_ckpt_data = infer_latent_dim_from_checkpoint(args.ae_ckpt)
            if args.ae_model != ae_model_type2 or args.ae_latent_dim != ae_latent_dim2 or args.ae_only_digit != ae_only_digit2:
                print(f"{emoji.error} Information in AE checkpoint {args.ae_ckpt} does not match with information stored in checkpoint: {resume_ckpt_path}") 
                exit(error_codes.MODEL_MISMATCH)
        else:
            args.ae_ckpt = ae_ckpt_from_resume_ckpt
            ae_ckpt_data = ae_ckpt_data_from_resume_ckpt
        
        print(f"{emoji.info} AE ckpt: {args.ae_ckpt}")
        print(f"{emoji.info} AE model: {args.ae_model}")
        print(f"{emoji.info} AE model latent dim: {args.ae_latent_dim}")
        print(f"{emoji.info} AE model only digit: {args.ae_only_digit}")
        
    #
    # Generate new experiment name if not resuming
    #
    if not exp_name:
        exp_name = Path(
            f"{args.model}"
            f"-E{args.ae_model}"
            f"-d{args.ae_only_digit if args.ae_only_digit is not None else 'all'}"
            f"-bs{args.batch_size}"
            f"-lr{args.lr:g}"
            f"-D{args.ae_latent_dim}"
            f"-T{args.T}"
            f"-seed{args.seed}"
            f"-{time.strftime('%Y%m%d-%H%M%S')}"
        )

    #
    # Output directories
    #
    ckpt_dir = outdir / "checkpoints" / exp_name
    sample_dir = outdir / "samples" / exp_name
    run_dir = outdir / "runs" / exp_name
    for d in (ckpt_dir, sample_dir, run_dir):
        d.mkdir(parents=True, exist_ok=True)

    #
    # Setting up logger
    #    
    log_level = logging.DEBUG if args.debug else logging.INFO
    train_logger = setup_component_logger("train", log_dir / f"{exp_name}.log", level=log_level)
    
    #
    # Printing some helpful information
    #
    train_logger.info(f"{emoji.info} EXPERIMENT: {exp_name}")
    train_logger.info(f"{emoji.info} MODEL: {args.model}")
    train_logger.debug(f"ckpt_dir={ckpt_dir}")
    train_logger.debug(f"sample_dir={sample_dir}")
    train_logger.debug(f"run_dir={run_dir}")
    train_logger.debug(f"log_dir={log_dir}")

    log_args_rich(train_logger, args, file_only=True)

    device = get_device(train_logger, args.device)
    dl_gen = seed_everything(args.seed, True)
    install_signal_handlers()

    train_logger.info(f"Using autoencoder checkpoint: {args.ae_ckpt}")
    train_logger.info(f"Detected autoencoder type: {args.ae_model}")
    train_logger.info(f"Inferred latent_dim={args.ae_latent_dim} from autoencoder checkpoint")

    # Load data
    train_logger.info(f"Loading dataset")
    train_logger.info(f'- Digit used: {args.ae_only_digit}')
    mnist_dataset = HF_MNIST(split="train", only_digit=args.ae_only_digit, cache_dir=args.cache_dir)
    train_logger.info(f"- Cache directory: {args.cache_dir}")
    mnist_loader, n_train = make_mnist_loader(
        mnist_dataset=mnist_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=dl_gen,
        worker_init_fn=seed_worker
    )
    train_logger.info(f"- Dataset ready (images: {len(mnist_dataset)})")

    # Load pretrained autoencoder
    train_logger.info(f"Loading pretrained autoencoder from {args.ae_ckpt}")
    encoder, decoder = load_autoencoder(args.ae_model, args.ae_latent_dim, args.ae_only_digit, ae_ckpt_data, device, train_logger)
    train_logger.info(f"- Autoencoder loaded successfully")

    # Model, optimizer, schedules
    net = LatentDenoiser(
        latent_dim=args.ae_latent_dim,
        hidden_dim=args.hidden_dim,
        time_dim=256
    ).to(device)
    train_logger.info(f"Initialized latent denoiser (latent_dim={args.ae_latent_dim}, hidden_dim={args.hidden_dim})")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = precompute_schedules(args.T, args.beta_start, args.beta_end, device)

    # AMP setup
    use_amp = bool(args.amp and device in ("cuda", "mps", "cpu"))
    amp_dtype = torch.float16 if device in ("cuda", "mps") else torch.bfloat16
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp)
        if use_amp else nullcontext()
    )
    if device == "cuda" and use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        class _NullScaler:
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass
            def state_dict(self): return {}
            def load_state_dict(self, sd): pass
        scaler = _NullScaler()

    # Load checkpoint if resuming
    start_epoch = 1
    global_step = 0
    if resume_ckpt_path and resume_ckpt_path.is_file():
        ep, global_step, _ = load_checkpoint(resume_ckpt_path, net, opt, scaler, map_location=device)
        start_epoch = ep + 1
        train_logger.info(f"[Resume] {resume_ckpt_path} (epoch={ep}, global_step={global_step})")

    if start_epoch > args.epochs:
        train_logger.info(f"[Resume/Auto-Resume] Already trained. Exiting.")
        return

    # TensorBoard
    writer = SummaryWriter(log_dir=str(run_dir))
    train_logger.info(f"TensorBoard logs at: {writer.log_dir}")

    # Training loop
    last_ckpt_time = time.time()
    last_ckpt_epoch = None
    best_loss = float("inf")
    best_ckpt_path = None
    epochs_without_improvement = 0
    train_logger.info(f"Training for {args.epochs-start_epoch+1} epochs")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        net.train()
        running = 0.0

        for x0, labels in mnist_loader:
            x0 = x0.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            B = x0.size(0)

            # Encode to latent space
            with torch.no_grad():
                z0 = encoder(x0)  # (B, latent_dim)

            # Sample timesteps
            t = torch.randint(0, args.T, (B,), device=device, dtype=torch.long)

            # Forward diffusion + predict noise
            with autocast_ctx:
                z_t, eps = q_sample(z0, t, sched)
                eps_hat = net(z_t, t, labels)  # Pass labels for conditional training
                loss = F.mse_loss(eps_hat, eps)

            # Backward pass
            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            running += loss.item() * B
            global_step += 1

        epoch_loss = running / n_train

        # SLURM timeout check
        if stop_requested(args.outdir):
            ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args, exp_name)
            train_logger.info(f"[Checkpoint] {ckpt_path}")
            train_logger.info("Timeout")
            exit(-1)

        # Generate samples
        if args.generate_images:
            with torch.no_grad():
                # Generate samples with specific labels (e.g., 2 samples per digit for 10 digits = 20 samples)
                # Or generate samples_per_epoch samples with labels cycling through 0-9
                samples_labels = torch.arange(args.samples_per_epoch, device=device) % 10
                imgs = generate_images(net, decoder, sched, n=args.samples_per_epoch,
                                       latent_dim=args.ae_latent_dim, labels=samples_labels, device=device).cpu()
            grid = tvutils.make_grid(imgs, nrow=int(args.samples_per_epoch ** 0.5), padding=2)
            tvutils.save_image(grid, sample_dir / f"epoch_{epoch:06d}.png")
            writer.add_image("samples/grid", grid, global_step=epoch)

        # Logging
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

        # Checkpointing
        now = time.time()
        if (epoch % args.ckpt_save_every == 0 or now - last_ckpt_time >= args.ckpt_every_sec):
            ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args, exp_name)
            train_logger.info(f"[Checkpoint] {ckpt_path}")
            last_ckpt_time = now
            last_ckpt_epoch = epoch

        # Track best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_without_improvement = 0
            if last_ckpt_epoch != epoch:
                best_ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args, exp_name)
                train_logger.info(f"[Checkpoint] {best_ckpt_path}")
                last_ckpt_epoch = epoch
            else:
                best_ckpt_path = ckpt_path
        else:
            epochs_without_improvement += 1

        # Log epoch
        elapsed = time.time() - epoch_start
        train_logger.info(f"[Epoch {epoch}/{args.epochs}] loss={epoch_loss:.6f} elapsed={elapsed:.1f}s "
                          f"(best_loss={best_loss:.6f} best={best_ckpt_path})")

        # Early stopping check
        if args.early_stopping_patience is not None and epochs_without_improvement >= args.early_stopping_patience:
            train_logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break

    # Final checkpoint
    if last_ckpt_epoch != epoch:
        ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args, exp_name)
        train_logger.info(f"[Checkpoint] {ckpt_path}")

    # Log hyperparameters
    writer.add_hparams(
        {'latent_dim': args.ae_latent_dim, 'hidden_dim': args.hidden_dim,
         'T': args.T, 'bs': args.batch_size, 'lr': args.lr, 'seed': args.seed},
        {'hparams/best_loss': best_loss}
    )
    writer.flush()
    writer.close()
    train_logger.info(f"Done")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
