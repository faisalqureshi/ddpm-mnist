from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, argparse, time, random, signal
from typing import Optional
import torch
import torch.nn.functional as F
from torchvision import utils as tvutils
from torch.utils.tensorboard import SummaryWriter
from common.device import get_device, seed_everything, seed_worker, make_output_directories
from common.data import HF_MNIST, make_mnist_loader, split_train_val
from common.logging import setup_component_logger, log_args_rich
from common.slurm import install_signal_handlers, stop_requested
from common.ckpt import load_checkpoint, find_latest_checkpoint, save_checkpoint_and_link_latest, find_latest_experiment
from torch.utils.data import DataLoader
from contextlib import nullcontext
import model
import model_conv
import torch.nn as nn
import logging

def build_model(model_str, latent_dim, device):
    if model_str == "conv":
        net = model_conv.Autoencoder(latent_dim = latent_dim).to(device)
    elif model_str == "mlp":
        net = model.Autoencoder(latent_dim = latent_dim).to(device)
    else:
        return None

    net.initialize_weights()
    return net

@torch.no_grad()
def evaluate(net, loader, device, autocast_ctx):
    net.eval()
    n_samples = 0
    running_loss = 0.0
    for x, _ in loader:
        x = x.to(device, dtype=torch.float32)
        B = x.size(0)

        with autocast_ctx:
            x_hat, _ = net(x)
            loss = F.binary_cross_entropy(x_hat, x)  # BCE for Sigmoid output

        n_samples += B
        running_loss += loss.item() * B
    return running_loss / n_samples

def main():
    parser = argparse.ArgumentParser("MNIST Autoencoder")
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("SLURM_TMPDIR", "./hf_cache"),
                        help="HuggingFace cache directory")
    parser.add_argument("--model", type=str, default="mlp", help="'mlp' or 'conv'")
    parser.add_argument("--outdir", type=str, default=os.environ.get("SCRATCH", "./outputs"))
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--only-digit", type=int, default=None, help="Use only this digit")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Early stopping patience (epochs)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ckpt-every-sec", type=int, default=300, help="Save checkpoint after elapsed second")
    parser.add_argument("--ckpt-save-every", type=int, default=1, help="Save checkpoint every epoch")
    parser.add_argument("--resume", type=str, default=None, help="Name of a checkpoint to resume training from.")
    parser.add_argument("--auto-resume", action="store_true", default=False, help="Resume from the most recent checkpoint.")
    parser.add_argument("--generate-images", action="store_true", default=False, help="Generate sample images at each epoch.")
    parser.add_argument("--debug", action="store_true", default=False, help="Turn on debugging messages.")
    args = parser.parse_args()

    #
    # Resume logic
    #
    exp_name = None
    resume = False
    auto_resume = False
    if args.resume:
        resume_ckpt_path = Path(args.resume)
        resume = True if resume_ckpt_path.is_file() else False
        exp_name = resume_ckpt_path.parent.name
    elif args.auto_resume:
        f = find_latest_experiment(Path(args.outdir), args.model)
        if f:
            exp_name = f.name
            auto_resume = True if f.is_dir() else False
    
    if not exp_name:    
        exp_name = Path(
            f"{args.model}"
            f"-d{args.only_digit if args.only_digit is not None else 'all'}"
            f"-bs{args.batch_size}"
            f"-lr{args.lr:g}"
            f"-D{args.latent_dim}"
            f"-seed{args.seed}"
            f"-{time.strftime('%Y%m%d-%H%M%S')}"
        )

    #
    # Output directories
    #
    ckpt_dir, sample_dir, run_dir, log_dir = make_output_directories(args.outdir, exp_name)
    log_level = logging.DEBUG if args.debug else logging.INFO
    train_logger = setup_component_logger("train", log_dir / f"{exp_name}.log", level=log_level)
    train_logger.info(f"EXPERIMENT: {exp_name}")
    train_logger.debug(f"ckpt_dir={ckpt_dir}")
    train_logger.debug(f"sample_dir={sample_dir}")
    train_logger.debug(f"run_dir={run_dir}")
    train_logger.debug(f"log_dir={log_dir}")

    #
    # Commandline arguments
    #
    log_args_rich(train_logger, args, file_only=True)

    #
    # Pick device, setup random seed, and install slurm handlers
    #
    device = get_device(train_logger, args.device)
    dl_gen = seed_everything(args.seed, True) 
    install_signal_handlers()

    #
    # Data
    #
    train_logger.info(f"Loading dataset")
    train_logger.info(f'- Digit used: {args.only_digit}')
    train_logger.info(f"- Cache directory: {args.cache_dir}")
    mnist_dataset = HF_MNIST(split="train",
                             only_digit=args.only_digit,
                             cache_dir=args.cache_dir)
    train_ds, val_ds = split_train_val(mnist_dataset, 
                                       val_fraction=0.1, 
                                       seed=args.seed)
    train_loader, n_train = make_mnist_loader(mnist_dataset=train_ds, 
                                              batch_size=args.batch_size, 
                                              num_workers=args.num_workers, 
                                              generator=dl_gen, 
                                              worker_init_fn=seed_worker)
    val_loader, n_val = make_mnist_loader(mnist_dataset=val_ds, 
                                          batch_size=args.batch_size, 
                                          num_workers=args.num_workers, 
                                          generator=dl_gen, 
                                          worker_init_fn=seed_worker)
    train_logger.info(f"- Dataset ready (train: {n_train}, val: {n_val})")

    #
    # Model, optimizer and compute settings
    #
    net = build_model(args.model, args.latent_dim, device)
    if not net:
        train_logger.error(f"Cannot initialize model: {args.model}.  Exiting.")
        exit(-2)
    else:
        train_logger.info(f"Initialized model: {args.model}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    # Learning rate scheduler
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.epochs // 3, gamma=0.1)
    else:
        scheduler = None

    use_amp = bool(args.amp and device in ("cuda", "mps", "cpu"))
    amp_dtype = torch.float16 if device in ("cuda", "mps") else torch.bfloat16
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp)
        if use_amp else nullcontext()
    )
    if device == "cuda" and use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:  # very old PyTorch
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        class _NullScaler:
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass
            def state_dict(self): return {}
            def load_state_dict(self, sd): pass
        scaler = _NullScaler()

    #
    # Resume logic
    #
    start_epoch = 1
    global_step = 0
    if resume:
        ep, global_step, _ = load_checkpoint(resume_ckpt_path, net, opt, scaler, map_location=device)
        start_epoch = ep + 1
        train_logger.info(f"[Resume] {resume_ckpt_path} (epoch={ep}, global_step={global_step})")
    elif auto_resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            ep, global_step, _ = load_checkpoint(latest, net, opt, scaler, map_location=device)
            start_epoch = ep + 1
            train_logger.info(f"[Auto-Resume] {latest} (epoch={ep}, global_step={global_step})")

    if start_epoch > args.epochs:
        train_logger.info(f"[Resume/Auto-Resume] (start_epoch={start_epoch} >= total_epochs={args.epochs}).  Exiting.")
        return

    #
    # Samples
    #
    if args.generate_images:
        N_SAMPLES = 16
        sample_loader = DataLoader(train_ds, 
                                   batch_size=N_SAMPLES, 
                                   shuffle=True, 
                                   num_workers=0, 
                                   generator=dl_gen, 
                                   worker_init_fn=seed_worker)
        sample_x, _ = next(iter(sample_loader))
        sample_x = sample_x.to(device)
        grid = tvutils.make_grid(sample_x, nrow=int(N_SAMPLES ** 0.5), padding=2)
        tvutils.save_image(grid, sample_dir / f"sample_x.png")

    #
    # TensorBoard writer
    #
    writer = SummaryWriter(log_dir=str(run_dir))
    train_logger.info(f"TensorBoard logs at: {writer.log_dir}")

    #
    # Epochs - start
    #
    train_logger.info(f"Training for {args.epochs-start_epoch+1} epochs")
    last_ckpt_time = time.time()
    last_ckpt_epoch = None
    best_val = float("inf")
    best_ckpt_path = None
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        #
        # Single epoch - start
        #
        net.train()
        n_samples = 0
        running_loss = 0.0
        latent_means = []
        latent_stds = []

        for x, _ in train_loader:
            x = x.to(device, dtype=torch.float32)
            B = x.size(0)

            with autocast_ctx:
                x_hat, z = net(x)
                loss = F.binary_cross_entropy(x_hat, x)  # BCE for Sigmoid output

            # Collect latent statistics
            latent_means.append(z.mean().item())
            latent_stds.append(z.std().item())

            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.grad_clip)
                opt.step()

            n_samples += B
            running_loss += loss.item() * B
            global_step += 1

        #
        # Single epoch - end
        #
        epoch_loss = running_loss / n_samples

        #
        # Aborting on slurm signal
        #
        if stop_requested(args.outdir):
            ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args, exp_name)
            train_logger.info(f"[Checkpoint] {ckpt_path}")
            train_logger.info("Timeout")
            exit(-1)
        
        #
        # Validation loss
        #
        val_loss = evaluate(net, val_loader, device, autocast_ctx)

        #
        # Sample generation
        #
        if args.generate_images:
            net.eval()
            with torch.no_grad():
                sample_x_hat, _ = net(sample_x)
                grid = tvutils.make_grid(sample_x_hat, nrow=int(N_SAMPLES ** 0.5), padding=2)
                tvutils.save_image(grid, sample_dir / f"epoch_{epoch:06d}.png")
                writer.add_image("samples/grid", grid, global_step=epoch)

        #
        # Logging at TensorBoard
        #
        writer.add_scalars(
            "loss/epoch",               # one chart
            {"train": epoch_loss, "val": val_loss},
            global_step=epoch
        )

        # Log latent statistics
        avg_latent_mean = sum(latent_means) / len(latent_means)
        avg_latent_std = sum(latent_stds) / len(latent_stds)
        writer.add_scalar("latent/mean", avg_latent_mean, global_step=epoch)
        writer.add_scalar("latent/std", avg_latent_std, global_step=epoch)

        # Log learning rate
        current_lr = opt.param_groups[0]['lr']
        writer.add_scalar("train/lr", current_lr, global_step=epoch)

        #
        # Checkpointing
        #
        now = time.time()
        if (epoch % args.ckpt_save_every == 0 or now - last_ckpt_time >= args.ckpt_every_sec):
            ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args, exp_name)
            train_logger.info(f"[Checkpoint] {ckpt_path}")
            last_ckpt_time = now
            last_ckpt_epoch = epoch

        #
        # Keeping the best model
        #
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            if last_ckpt_epoch != epoch:
                best_ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args, exp_name)
                train_logger.info(f"[Checkpoint] {best_ckpt_path}")
                last_ckpt_epoch = epoch
            else:
                best_ckpt_path = ckpt_path
        else:
            epochs_without_improvement += 1

        #
        # Learning rate scheduler step
        #
        if scheduler is not None:
            scheduler.step()

        #
        # Logging each epoch
        #
        elapsed = time.time() - epoch_start
        train_logger.info(f"[Epoch {epoch}/{args.epochs}] "
                          f"train_loss={epoch_loss:.6f} val_loss={val_loss:.6f} "
                          f"lr={current_lr:.2e} "
                          f"elapsed={elapsed:.1f}s "
                          f"(best_val_loss={best_val:.6f} best={best_ckpt_path})")

        #
        # Early stopping check
        #
        if args.early_stopping_patience is not None and epochs_without_improvement >= args.early_stopping_patience:
            train_logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break

    #
    # Epochs - done
    #

    #
    # Saving checkpoint
    #
    if last_ckpt_epoch != epoch:
        ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args, exp_name)
        train_logger.info(f"[Checkpoint] {ckpt_path}")

    #
    # Logging hyperparameters to tensorboard, and closing writer
    #
    writer.add_hparams(
        {'model': args.model, 'digit': str(args.only_digit), 'bs': args.batch_size,
        'lr': args.lr, 'lr_scheduler': args.lr_scheduler, 'seed': args.seed, 'amp': int(args.amp)},
        {'hparams/best_val_loss': best_val}  # Fixed: use best_val instead of val_loss
    )
    writer.flush()
    writer.close()
    train_logger.info(f"Done")

#
# Entry main
#
if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()