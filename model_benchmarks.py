#!/usr/bin/env python3
"""
Model benchmarks: RAM consumption, model size, inference speed, and power estimation.

Usage:
    python model_benchmarks.py
    python model_benchmarks.py --pretrained model/pretrained_encoder.pth
    python model_benchmarks.py --batch-sizes 1 4 8
"""
import argparse
import os
import time
import tracemalloc

import numpy as np
import torch

from train import ConvTransformerModel
from data_segmentation_helpers.config import (
    CONV_EMBEDDING_DIM,
    SEGMENT_DURATION,
    SEQUENCE_BATCH_SIZE,
    SEQUENCE_LENGTH,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_FFN_DIM,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS,
    USE_CLS_TOKEN,
)


def create_model():
    return ConvTransformerModel(
        num_input_channels=18,
        num_classes=2,
        sequence_length=SEQUENCE_LENGTH,
        embed_dim=CONV_EMBEDDING_DIM,
        num_layers=TRANSFORMER_NUM_LAYERS,
        num_heads=TRANSFORMER_NUM_HEADS,
        ffn_dim=TRANSFORMER_FFN_DIM,
        dropout=TRANSFORMER_DROPOUT,
        use_cls_token=USE_CLS_TOKEN,
    )


def format_bytes(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def flops_analysis(model):
    """Estimate FLOPs per inference (hardware-independent computational cost)."""
    print(f"\n{'=' * 70}")
    print("COMPUTATIONAL COST (FLOPs)")
    print(f"{'=' * 70}")

    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.randn(1, SEQUENCE_LENGTH, 18, 128, 9)
        model.eval()
        with FlopCounterMode(display=False) as fcm:
            with torch.no_grad():
                _ = model(x)
        total_flops = fcm.get_total_flops()
        print(f"Total FLOPs (1 sample):  {total_flops:,.0f}")
        print(f"                         {total_flops/1e6:.2f} MFLOPs")
        print(f"                         {total_flops/1e9:.4f} GFLOPs")
        del x
        return total_flops
    except ImportError:
        pass

    # Fallback: manual estimation
    print("(torch.utils.flop_counter not available, using manual estimation)")

    # Conv tower: per segment
    # Block1: Conv2d(18->32, k=3) on (128,9): 18*32*3*3*128*9 * 2 MACs
    conv1 = 18 * 32 * 3 * 3 * 128 * 9 * 2
    # Block2: Conv2d(32->64, k=3) on (64,4): after maxpool
    conv2 = 32 * 64 * 3 * 3 * 64 * 4 * 2
    # Block3: Conv2d(64->128, k=3) on (32,2): after maxpool
    conv3 = 64 * 128 * 3 * 3 * 32 * 2 * 2
    conv_per_seg = conv1 + conv2 + conv3
    conv_total = conv_per_seg * SEQUENCE_LENGTH

    # Transformer: self-attention + FFN per layer
    d = CONV_EMBEDDING_DIM
    seq = SEQUENCE_LENGTH + (1 if USE_CLS_TOKEN else 0)
    # Self-attention: 4 * seq * d^2 (Q,K,V,O projections) + 2 * seq^2 * d (attention)
    attn_per_layer = 4 * seq * d * d * 2 + 2 * seq * seq * d * 2
    # FFN: 2 * seq * d * ffn_dim
    ffn_per_layer = 2 * seq * d * TRANSFORMER_FFN_DIM * 2
    transformer_total = (attn_per_layer + ffn_per_layer) * TRANSFORMER_NUM_LAYERS

    # FC head
    fc_total = d * 64 * 2 + 64 * 2 * 2

    total = conv_total + transformer_total + fc_total
    print(f"Conv tower:    {conv_total:>14,} FLOPs ({conv_total/1e6:.1f}M)")
    print(f"Transformer:   {transformer_total:>14,} FLOPs ({transformer_total/1e6:.1f}M)")
    print(f"FC head:       {fc_total:>14,} FLOPs ({fc_total/1e6:.3f}M)")
    print(f"Total:         {total:>14,} FLOPs ({total/1e6:.1f}M / {total/1e9:.4f}G)")
    return total


def model_size_analysis(model):
    """Analyze model size: parameters, disk size, per-layer breakdown."""
    print("=" * 70)
    print("MODEL SIZE ANALYSIS")
    print("=" * 70)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Size in memory (float32 = 4 bytes per param)
    param_bytes = total_params * 4
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter memory:     {format_bytes(param_bytes)}")
    print(f"Buffer memory:        {format_bytes(buffer_bytes)}")
    print(f"Total model size:     {format_bytes(total_bytes)}")

    # Save and check disk size
    tmp_path = "/tmp/model_size_test.pth"
    torch.save(model.state_dict(), tmp_path)
    disk_size = os.path.getsize(tmp_path)
    print(f"Disk size (.pth):     {format_bytes(disk_size)}")
    os.remove(tmp_path)

    # Per-component breakdown
    print(f"\nPer-component breakdown:")
    print(f"  {'Component':<30} {'Params':>12} {'Size':>12}")
    print(f"  {'-'*54}")

    components = {
        "conv_tower": model.conv_tower,
        "transformer": model.transformer,
        "norm": model.norm,
        "fc": model.fc,
    }
    if hasattr(model, "cls_token") and model.cls_token is not None:
        # CLS token is a parameter, not a module
        cls_params = model.cls_token.numel()
        print(f"  {'cls_token':<30} {cls_params:>12,} {format_bytes(cls_params * 4):>12}")

    if hasattr(model, "pos_embedding"):
        pos_params = model.pos_embedding.numel()
        print(f"  {'pos_embedding':<30} {pos_params:>12,} {format_bytes(pos_params * 4):>12}")

    for name, module in components.items():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<30} {n:>12,} {format_bytes(n * 4):>12}")

    return total_params, total_bytes


def ram_analysis(model, batch_sizes, device):
    """Measure peak RAM during inference."""
    print(f"\n{'=' * 70}")
    print("RAM CONSUMPTION (INFERENCE)")
    print(f"{'=' * 70}")

    input_shape_info = f"(B, {SEQUENCE_LENGTH}, 18, 128, 9)"
    input_elem_size = SEQUENCE_LENGTH * 18 * 128 * 9 * 4  # float32
    print(f"Input tensor shape: {input_shape_info}")
    print(f"Single sample size: {format_bytes(input_elem_size)}")

    print(f"\n  {'Batch':>6} {'Input':>12} {'Peak RAM':>12} {'RAM/sample':>12} {'GPU mem':>12}")
    print(f"  {'-'*56}")

    for bs in batch_sizes:
        # CPU RAM measurement with tracemalloc
        tracemalloc.start()

        model_cpu = create_model()
        model_cpu.load_state_dict(model.state_dict())
        model_cpu.eval()

        x = torch.randn(bs, SEQUENCE_LENGTH, 18, 128, 9)
        input_bytes = x.nelement() * x.element_size()

        with torch.no_grad():
            _ = model_cpu(x)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # GPU memory (if applicable)
        gpu_mem = "N/A"
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            model_gpu = create_model().to(device)
            model_gpu.load_state_dict(model.state_dict())
            model_gpu.eval()
            x_gpu = torch.randn(bs, SEQUENCE_LENGTH, 18, 128, 9, device=device)
            with torch.no_grad():
                _ = model_gpu(x_gpu)
            gpu_mem = format_bytes(torch.cuda.max_memory_allocated())
            del model_gpu, x_gpu
            torch.cuda.empty_cache()

        print(
            f"  {bs:>6} {format_bytes(input_bytes):>12} "
            f"{format_bytes(peak):>12} {format_bytes(peak // bs):>12} {gpu_mem:>12}"
        )

        del model_cpu, x


def inference_speed(model, device, batch_sizes, n_warmup=3, n_runs=20):
    """Measure inference latency and throughput."""
    print(f"\n{'=' * 70}")
    print(f"INFERENCE SPEED (device: {device})")
    print(f"{'=' * 70}")

    seq_duration = SEQUENCE_LENGTH * SEGMENT_DURATION  # seconds of EEG per sample
    print(f"Each sample = {seq_duration}s of EEG ({SEQUENCE_LENGTH} segments x {SEGMENT_DURATION}s)")

    print(f"\n  {'Batch':>6} {'Latency':>12} {'Per sample':>12} {'Throughput':>14} {'RT ratio':>10}")
    print(f"  {'-'*58}")

    model.to(device)
    model.eval()

    for bs in batch_sizes:
        x = torch.randn(bs, SEQUENCE_LENGTH, 18, 128, 9, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        # Timed runs
        times = []
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            times.append(time.perf_counter() - t0)

        mean_t = np.mean(times)
        std_t = np.std(times)
        per_sample = mean_t / bs
        throughput = bs / mean_t
        # Real-time ratio: how much faster than real-time EEG
        rt_ratio = seq_duration / per_sample

        print(
            f"  {bs:>6} {mean_t*1000:>8.1f}ms +/-{std_t*1000:>4.1f} "
            f"{per_sample*1000:>8.1f}ms {throughput:>10.1f} samp/s {rt_ratio:>8.0f}x"
        )

        del x


def power_estimation(model, device, duration=30):
    """Estimate power consumption during sustained inference."""
    print(f"\n{'=' * 70}")
    print("POWER CONSUMPTION ESTIMATE")
    print(f"{'=' * 70}")

    if device.type == "cuda":
        # NVIDIA GPUs: use nvidia-smi
        try:
            import subprocess

            # Idle power
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            idle_power = float(result.stdout.strip().split("\n")[0])

            # Load power: run inference for `duration` seconds
            model.to(device)
            model.eval()
            x = torch.randn(1, SEQUENCE_LENGTH, 18, 128, 9, device=device)

            end_time = time.time() + duration
            power_readings = []
            while time.time() < end_time:
                with torch.no_grad():
                    _ = model(x)
                torch.cuda.synchronize()
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True,
                )
                power_readings.append(float(result.stdout.strip().split("\n")[0]))

            avg_power = np.mean(power_readings)
            print(f"GPU idle power:      {idle_power:.1f} W")
            print(f"GPU inference power: {avg_power:.1f} W (avg over {duration}s)")
            print(f"Delta (model only):  {avg_power - idle_power:.1f} W")
            del x
        except Exception as e:
            print(f"Could not measure GPU power: {e}")

    elif device.type == "mps":
        # Apple Silicon: use powermetrics (requires sudo) or estimate
        print("Apple Silicon detected.")
        print("For precise measurements, run (requires sudo):")
        print(f"  sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n {duration}")
        print("  (while this script runs inference in another terminal)")
        print()

        # Run sustained inference and report timing
        model.to(device)
        model.eval()
        x = torch.randn(1, SEQUENCE_LENGTH, 18, 128, 9, device=device)

        n_iters = 0
        t0 = time.time()
        while time.time() - t0 < duration:
            with torch.no_grad():
                _ = model(x)
            torch.mps.synchronize()
            n_iters += 1

        elapsed = time.time() - t0
        print(f"Sustained inference: {n_iters} iterations in {elapsed:.1f}s ({n_iters/elapsed:.1f} it/s)")
        print(f"Typical Apple Silicon GPU power for this workload: ~5-15W (estimated)")
        print(f"Energy per inference: ~{(10 / (n_iters/elapsed)):.2f} Watt-seconds (estimated at 10W)")
        del x

    else:
        print("CPU mode — power measurement not directly available.")
        print("Typical CPU inference power: 15-65W depending on hardware.")

    # Energy per prediction
    print(f"\nFor reference:")
    print(f"  Each prediction covers {SEQUENCE_LENGTH * SEGMENT_DURATION / 60:.0f} minutes of EEG")
    print(f"  Continuous monitoring would need ~{3600 / (SEQUENCE_LENGTH * SEGMENT_DURATION):.1f} predictions/hour")


def main():
    parser = argparse.ArgumentParser(description="Model benchmarks")
    parser.add_argument("--pretrained", type=str, default="model/pretrained_encoder.pth")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model = create_model()
    if os.path.exists(args.pretrained):
        model.load_state_dict(
            torch.load(args.pretrained, map_location="cpu", weights_only=False)
        )
        print(f"Loaded weights: {args.pretrained}")
    else:
        print("No pretrained weights found, using random init")

    print(f"Device: {device}")
    print(f"Config: seq_len={SEQUENCE_LENGTH}, embed_dim={CONV_EMBEDDING_DIM}, "
          f"layers={TRANSFORMER_NUM_LAYERS}, heads={TRANSFORMER_NUM_HEADS}, "
          f"cls_token={USE_CLS_TOKEN}")

    # Run all benchmarks
    model_size_analysis(model)
    flops_analysis(model)
    ram_analysis(model, args.batch_sizes, device)
    inference_speed(model, device, args.batch_sizes)
    power_estimation(model, device, duration=15)

    print(f"\n{'=' * 70}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
