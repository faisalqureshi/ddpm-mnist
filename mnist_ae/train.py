from pathlib import Path
import sys
sys.path.append(str(Path.cwd() / "../mnist_single_digit" ))

import os, argparse, time, random, signal
from typing import Optional
import torch
import torch.nn.functional as F
from torchvision import utils as tvutils
from torch.utils.tensorboard import SummaryWriter
from util import get_device, seed_everything, seed_worker, make_output_directories
from data import HF_MNIST, make_mnist_loader, split_train_val
from logging_setup import setup_component_logger, log_args_rich
from slurm_stop import install_signal_handlers, stop_requested
from ckpt import load_checkpoint, find_latest_checkpoint, save_checkpoint_and_link_latest
from torch.utils.data import DataLoader
from contextlib import nullcontext
import model
import model_conv
import torch.nn as nn

def build_model(model_str, device):
    if model_str == "conv":
        net = model_conv.Autoencoder().to(device)
    elif model_str == "mlp":
        net = model.Autoencoder().to(device)
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
            loss = F.mse_loss(x_hat, x)  # sum to average later
        
        n_samples += B
        running_loss += loss.item() * B
    return running_loss / n_samples

def main():
    parser = argparse.ArgumentParser("DDPM MNIST trainer (SHARCNET-ready)")
    parser.add_argument("--data-root", type=str, default=os.environ.get("SLURM_TMPDIR", "./hf_cache"))
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--outdir", type=str, default=os.environ.get("SCRATCH", "./outputs"))
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--only-digit", type=int, default=None, help="Use only this digit")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ckpt-every-sec", type=int, default=300, help="Save checkpoint after elapsed second")
    parser.add_argument("--ckpt-save-every", type=int, default=1, help="Save checkpoint every epoch")
    parser.add_argument("--resume", type=str, default=None, help="Name of a checkpoint to resume training from.")
    parser.add_argument("--auto-resume", action="store_true", default=False, help="Resume from the most recent checkpoint.")
    parser.add_argument("--generate-images", action="store_true", default=False, help="Generate sample images at each epoch.")
    args = parser.parse_args()

    #
    # Output directories
    #
    exp_name = Path(
        f"{args.model}"
        f"-d{args.only_digit if args.only_digit is not None else 'all'}"
        f"-bs{args.batch_size}"
        f"-lr{args.lr:g}"
        f"-seed{args.seed}"
        f"-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    ckpt_dir, sample_dir, run_dir, log_dir = make_output_directories(args.outdir, exp_name)
    train_logger = setup_component_logger("train", log_dir / f"{exp_name}.log")

    #
    # Commandline arguments
    #
    train_logger.info("=== Commandline args ===")
    log_args_rich(train_logger, args)

    #
    # Pick device, setup random seed, and install slurm handlers
    #
    device = get_device(train_logger, args.device)
    dl_gen = seed_everything(args.seed, True) 
    install_signal_handlers()

    #
    # Data
    #
    train_logger.info(f"=== Loading dataset ===")
    train_logger.info(f'- Digit used: {args.only_digit}')
    train_logger.info(f"- Cache directory: {args.data_root}")
    mnist_dataset = HF_MNIST(split="train", 
                             only_digit=args.only_digit, 
                             cache_dir=args.data_root)
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
    net = build_model(args.model, device)
    if not net:
        train_logger.error(f"Cannot initialize model: {args.model}.  Exiting.")
        exit(-2)
    else:
        train_logger.info(f"Initialized model: {args.model}.")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

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
    if args.resume:
        path = Path(args.resume)
        ep, global_step, _ = load_checkpoint(path, net, opt, scaler, map_location=device)
        start_epoch = ep + 1
        train_logger.info(f"[Resume] {path} (epoch={ep}, global_step={global_step})")
    elif args.auto_resume:
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
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        #
        # Single epoch - start
        #
        net.train()
        n_samples = 0
        running_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device, dtype=torch.float32)
            B = x.size(0)

            with autocast_ctx:
                x_hat, _ = net(x)
                loss = F.mse_loss(x_hat, x)

            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
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
            ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args)
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

        #
        # Checkpointing
        #
        now = time.time()
        if (epoch % args.ckpt_save_every == 0 or now - last_ckpt_time >= args.ckpt_every_sec):
            ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args)
            train_logger.info(f"[Checkpoint] {ckpt_path}")
            last_ckpt_time = now
            last_ckpt_epoch = epoch

        #
        # Keeping the best model
        #
        if val_loss < best_val:
            best_val = val_loss
            if last_ckpt_epoch != epoch:
                best_ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args)
                train_logger.info(f"[Checkpoint] {best_ckpt_path}")
                last_ckpt_epoch = epoch
            else:
                best_ckpt_path = ckpt_path

        #
        # Logging each epoch
        #
        elapsed = time.time() - epoch_start
        train_logger.info(f"[Epoch {epoch}/{args.epochs}] "
                          f"train_loss={epoch_loss:.6f} val_loss={val_loss:.6f} "
                          f"elapsed={elapsed:.1f}s "
                          f"(best_val_loss={best_val:.6f} best={best_ckpt_path})")

    #
    # Epochs - done
    #

    #
    # Saving checkpoint
    #
    if last_ckpt_epoch != epoch:
        ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args)
        train_logger.info(f"[Checkpoint] {ckpt_path}")

    #
    # Logging hyperparameters to tensorboard, and closing writer
    #
    writer.add_hparams(
        {'model': args.model, 'digit': str(args.only_digit), 'bs': args.batch_size,
        'lr': args.lr, 'seed': args.seed, 'amp': int(args.amp)},
        {'hparams/best_val_loss': val_loss}  # or your tracked best
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