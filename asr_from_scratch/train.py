"""Train the Conformer with CTC loss under a fixed time budget.

Training length is set in minutes rather than epochs. The learning rate schedule
follows elapsed time, so a run always completes its full cycle and finishes on
schedule, however fast the GPUs turn out to be.

Two GPUs (DistributedDataParallel):
    torchrun --nproc_per_node=2 -m asr_from_scratch.train \
        --train data/train.100/manifest.jsonl data/train.360/manifest.jsonl \
        --val data/validation/manifest.jsonl --minutes 55

CPU smoke test on synthetic data:
    python -m asr_from_scratch.train --synthetic --minutes 1 --d-model 64 --layers 2 \
        --batch-seconds 20 --lr 2e-3 --workers 0
"""
from __future__ import annotations

import argparse
import math
import os
import time
from itertools import count
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .data import DurationBatchSampler, SpeechDataset, collate
from .evaluate import collect_log_probs, greedy_transcripts
from .features import SAMPLE_RATE
from .metrics import cer, wer
from .model import ConformerCTC, count_parameters

WARMUP_FRACTION = 0.1


def learning_rate(progress: float, peak: float) -> float:
    """Linear warm-up followed by a cosine decay, as a function of progress in [0, 1]."""
    if progress < WARMUP_FRACTION:
        return peak * max(progress / WARMUP_FRACTION, 1e-3)
    remaining = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
    return peak * max(0.5 * (1.0 + math.cos(math.pi * min(remaining, 1.0))), 1e-3)


def save_checkpoint(path: Path, **content) -> None:
    """Write through a temporary file so an interrupted session cannot corrupt it."""
    temporary = path.with_suffix(".tmp")
    torch.save(content, temporary)
    os.replace(temporary, path)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Conformer-CTC model.")
    parser.add_argument("--train", nargs="+", help="One or more training manifests")
    parser.add_argument("--val", help="Validation manifest")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke test)")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--minutes", type=float, default=55.0, help="Training time budget")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Use a fixed number of steps instead of the time budget")
    parser.add_argument("--batch-seconds", type=float, default=160.0,
                        help="Seconds of audio per batch on each GPU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--augment-after", type=float, default=0.25,
                        help="Fraction of the run before SpecAugment starts (1 disables it)")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="Skip longer clips")
    parser.add_argument("--d-model", type=int, default=144)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--conv-kernel", type=int, default=31)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2, help="Data loading workers per GPU")
    parser.add_argument("--eval-minutes", type=float, default=10.0, help="Time between evaluations")
    parser.add_argument("--val-clips", type=int, default=300, help="Clips used in each evaluation")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Continue from the last checkpoint")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_gpu = torch.cuda.is_available()
    if distributed:
        dist.init_process_group("nccl" if use_gpu else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if use_gpu:
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if use_gpu else "cpu"
    is_main = rank == 0

    args.output.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        from .synthetic import generate

        if is_main:
            generate(args.output / "synthetic_train", 600, seed=1)
            generate(args.output / "synthetic_val", 60, seed=2)
        if distributed:
            dist.barrier()
        args.train = [str(args.output / "synthetic_train" / "manifest.jsonl")]
        args.val = str(args.output / "synthetic_val" / "manifest.jsonl")
    if not args.train or not args.val:
        raise SystemExit("Provide --train and --val, or use --synthetic")

    train_set = SpeechDataset(args.train, max_seconds=args.max_seconds)
    val_set = SpeechDataset(args.val, max_seconds=40.0, max_clips=args.val_clips)
    sampler = DurationBatchSampler(train_set.durations, args.batch_seconds, rank, world_size)
    loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collate,
                        num_workers=args.workers, pin_memory=use_gpu,
                        persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None)

    model_config = {"d_model": args.d_model, "layers": args.layers, "heads": args.heads,
                    "conv_kernel": args.conv_kernel, "dropout": args.dropout}
    torch.manual_seed(0)  # identical initial weights on every GPU
    model = ConformerCTC(**model_config).to(device)
    torch.manual_seed(1 + rank)  # dropout and SpecAugment must differ across GPUs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_gpu)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    step, start_epoch, elapsed_before, best_cer = 0, 0, 0.0, float("inf")
    last_path, best_path = args.output / "last.pt", args.output / "best.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step, start_epoch = checkpoint["step"], checkpoint["epoch"] + 1
        elapsed_before, best_cer = checkpoint["elapsed"], checkpoint["best_cer"]
        if is_main:
            print(f"Resuming from step {step} ({elapsed_before / 60:.1f} min elapsed)", flush=True)

    if is_main:
        hours = sum(train_set.durations) / 3600
        print(f"Devices: {world_size} x {'GPU' if use_gpu else 'CPU'} | parameters: "
              f"{count_parameters(model) / 1e6:.1f} M | train: {len(train_set)} clips, {hours:.0f} h | "
              f"val: {len(val_set)} clips", flush=True)

    trained_model = model
    if distributed:
        trained_model = DistributedDataParallel(model, device_ids=[local_rank] if use_gpu else None)

    def validate(epoch: int, elapsed: float) -> None:
        nonlocal best_cer
        outputs, references = collect_log_probs(model, val_set, device, workers=min(args.workers, 1))
        hypotheses = greedy_transcripts(outputs)
        word_errors, score = wer(references, hypotheses), cer(references, hypotheses)
        print(f"validation | step {step} | {elapsed / 60:5.1f} min | WER {word_errors:.3f} | "
              f"CER {score:.3f}", flush=True)
        for reference, hypothesis in list(zip(references, hypotheses))[:2]:
            print(f"  ref: {reference}\n  hyp: {hypothesis}", flush=True)
        # CER rather than WER: an empty output scores a WER of exactly 1.0, which would
        # beat the early transcripts of a model that has actually started to learn.
        if score < best_cer:
            best_cer = score
            save_checkpoint(best_path, model=model.state_dict(), model_config=model_config,
                            step=step, wer=word_errors, cer=score)
        save_checkpoint(last_path, model=model.state_dict(), model_config=model_config,
                        optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        step=step, epoch=epoch, elapsed=elapsed, best_cer=best_cer)

    budget = args.minutes * 60
    eval_interval = args.eval_minutes * 60
    start_time = time.monotonic() - elapsed_before
    next_eval = (elapsed_before // eval_interval + 1) * eval_interval
    log_time, log_loss, log_steps, log_audio = time.monotonic(), 0.0, 0, 0.0
    trained_model.train()
    finished = False
    elapsed = elapsed_before
    epoch = start_epoch

    for epoch in count(start_epoch):
        sampler.set_epoch(epoch)
        for waves, wave_lengths, targets, target_lengths, _texts in loader:
            # Every process follows the clock of the first one, so all of them
            # apply the same learning rate and stop at exactly the same step.
            clock = torch.tensor([time.monotonic() - start_time], dtype=torch.float64, device=device)
            if distributed:
                dist.broadcast(clock, 0)
            elapsed = clock.item()
            progress = step / args.max_steps if args.max_steps else elapsed / budget
            if progress >= 1.0:
                finished = True
                break
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(progress, args.lr)
            # CTC models first have to escape a plateau in which they only emit blanks
            # and common letters. Hiding parts of the input during that phase can keep
            # them there, so augmentation starts once they are learning.
            model.use_augment = progress >= args.augment_after

            waves = waves.to(device, non_blocking=True)
            wave_lengths = wave_lengths.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                                enabled=use_gpu):
                log_probs, lengths = trained_model(waves, wave_lengths)
            # CTC runs in float32 for numerical stability.
            loss = ctc_loss(log_probs.float().transpose(0, 1), targets, lengths, target_lengths)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            log_loss += loss.item()
            log_steps += 1
            log_audio += wave_lengths.sum().item() / SAMPLE_RATE
            if is_main and step % args.log_every == 0:
                now = time.monotonic()
                print(f"step {step:6d} | epoch {epoch} | {100 * progress:5.1f}% | "
                      f"loss {log_loss / log_steps:7.3f} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                      f"{world_size * log_audio / (now - log_time):6.0f}x real time", flush=True)
                log_time, log_loss, log_steps, log_audio = now, 0.0, 0, 0.0

            if elapsed >= next_eval:
                next_eval += eval_interval
                if is_main:
                    validate(epoch, elapsed)
                    log_time, log_loss, log_steps, log_audio = time.monotonic(), 0.0, 0, 0.0
        if finished:
            break

    if is_main:
        validate(epoch, elapsed)
        print(f"Finished in {elapsed / 60:.1f} min, {step} steps, best validation CER {best_cer:.3f}",
              flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
