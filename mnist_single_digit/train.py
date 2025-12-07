#
# train.py
#

import os, argparse, time, random, signal
from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path for common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torchvision import utils as tvutils
from torch.utils.tensorboard import SummaryWriter
from model import Denoiser, precompute_schedules, q_sample, generate_images
from common.device import seed_everything, get_device
from common.data import HF_MNIST, make_mnist_loader
from common.logging import setup_component_logger, log_args_rich
from common.slurm import install_signal_handlers, stop_requested
from common.ckpt import load_checkpoint, find_latest_checkpoint, save_checkpoint_and_link_latest

def main():
    parser = argparse.ArgumentParser("DDPM MNIST trainer (SHARCNET-ready)")
    parser.add_argument("--data-root", type=str, default=os.environ.get("SLURM_TMPDIR", "./hf_cache"))
    parser.add_argument("--outdir", type=str, default=os.environ.get("SCRATCH", "./outputs"))
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--only-digit", type=int, default=5, help="-1 to use all digits")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=1000, help="Diffusion steps")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--samples-per-epoch", type=int, default=16, help="How many images to generated for viewing after each epoch.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ckpt-every-sec", type=int, default=300, help="Save checkpoint after elapsed second")
    parser.add_argument("--ckpt-save-every", type=int, default=1, help="Save checkpoint every epoch")
    parser.add_argument("--resume", type=str, default=None, help="Name of a checkpoint to resume training from.")
    parser.add_argument("--auto-resume", action="store_true", default=False, help="Resume from the most recent checkpoint.")
    parser.add_argument("--generate-images", action="store_true", default=False, help="Generate sample images at each epoch.")
    args = parser.parse_args()

    # but first, logging

    outdir = Path(args.outdir)
    log_dir = outdir / Path("logs") if args.logdir is None else Path(args.logdir)
    log_path = log_dir / "train.log"
    train_logger = setup_component_logger("train", log_path)
    train_logger.info(f"\nLogging to {log_path}")

    train_logger.info("=== Commandline args ===")
    log_args_rich(train_logger, args)

    device = get_device(train_logger, args.device)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    train_logger.info("=== Output folders ===")
    ckpt_dir = outdir / "checkpoints"
    sample_dir = outdir / "samples"
    run_dir = outdir / "runs" 
    for d in (ckpt_dir, sample_dir, run_dir):
        d.mkdir(parents=True, exist_ok=True)
        train_logger.info(f"- Created {d}")

    install_signal_handlers()

    # Data
    train_logger.info(f"=== Loading dataset ===")
    cache_dir = Path(args.data_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    only_digit = None if args.only_digit == -1 else args.only_digit
    train_logger.info(f'- Digit used: {only_digit}')
    mnist_dataset = HF_MNIST(split="train", only_digit=only_digit, cache_dir=args.data_root)
    train_logger.info(f"- Dataset cache directory: {args.data_root}")
    mnist_loader, n_train = make_mnist_loader(mnist_dataset=mnist_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    train_logger.info(f"- Dataset ready (images: {len(mnist_dataset)})")

    # Model/opt/sched
    net = Denoiser(base_ch=64, time_dim=256).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = precompute_schedules(args.T, args.beta_start, args.beta_end, device)

    use_amp = bool(args.amp and device.type == "cuda")
    try:
        # PyTorch >= 2.x unified AMP
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    except TypeError:
        # Older CUDA-only AMP
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    # Resume logic
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

    # TensorBoard
    writer = SummaryWriter(log_dir=str(run_dir))
    train_logger.info(f"TensorBoard logs at: {writer.log_dir}")

    # Checkpoints
    last_ckpt_time = time.time()
    last_ckpt_epoch = None

    train_logger.info(f"Training for {args.epochs-start_epoch+1} epochs")
    # Train
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        net.train()
        running = 0.0
        for x0, _ in mnist_loader:
            x0 = (x0.to(device) * 2) - 1
            B = x0.size(0)
            t = torch.randint(0, args.T, (B,), device=device, dtype=torch.long)

            with autocast_ctx():
                x_t, eps = q_sample(x0, t, sched)
                eps_hat = net(x_t, t)
                loss = F.mse_loss(eps_hat, eps)

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

        if args.generate_images:
            with torch.no_grad():
                imgs = generate_images(net, sched, n=args.samples_per_epoch, device=device).cpu()
            grid = tvutils.make_grid(imgs, nrow=int(args.samples_per_epoch ** 0.5), padding=2)
            tvutils.save_image(grid, sample_dir / f"epoch_{epoch:06d}.png")
            writer.add_image("samples/grid", grid, global_step=epoch)

        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        now = time.time()
        elapsed = now - epoch_start
        train_logger.info(f"[Epoch {epoch}/{args.epochs}] loss={epoch_loss:.6f} elapsed={elapsed} sec.")

        # Cooperative stop check
        if stop_requested(outdir):
            ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args, "single-digit")
            train_logger.info(f"[Checkpoint] {ckpt_path}")
            train_logger.info("Timeout")
            exit(-1)

        # Save checkpoint
        now = time.time()
        if (epoch % args.ckpt_save_every == 0 or now - last_ckpt_time >= args.ckpt_every_sec):
            ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args, "single-digit")
            train_logger.info(f"[Checkpoint] {ckpt_path}")
            last_ckpt_time = now
            last_ckpt_epoch = epoch

    # Final
    with torch.no_grad():
        imgs = generate_images(net, sched, n=args.samples_per_epoch, device=device).cpu()
    grid = tvutils.make_grid(imgs, nrow=int(args.samples_per_epoch ** 0.5), padding=2)
    tvutils.save_image(grid, sample_dir / f"sample_epoch_{epoch:06d}_final.png")

    if last_ckpt_epoch != epoch:
        ckpt_path = save_checkpoint_and_link_latest(ckpt_dir, net, opt, scaler, epoch, global_step, args, "single-digit")
        train_logger.info(f"[Checkpoint] {ckpt_path}")

    train_logger.info(f"Done")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
