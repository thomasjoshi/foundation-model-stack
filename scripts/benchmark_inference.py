import argparse
import logging
import os
import random
import statistics
import timeit
import threading
import time
import csv

import numpy as np
import torch
from torch import distributed as dist
from torch._dynamo import OptimizedModule

from fms import models
from fms.utils import fusion, generation, print0, tokenizers


# Example running llama 7B on one A100:
#
# (bare metal) $ CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 ./scripts/benchmark_inference.py --architecture=llama --variant=7b --tokenizer=~/models/tokenizer.model --batch_size=2 --seq_len=500
# (slurm) $ srun -N 1 --gres=gpu:1 torchrun --nproc_per_node=1 ./scripts/benchmark_inference.py --architecture=llama --variant=7b --tokenizer=~/models/tokenizer.model --batch_size=2 --seq_len=500
# loading model
# loading complete on rank 0
# Uncompiled results:
# - with use_cache=True
#         34.86 ms per token
# - with use_cache=False
#         86.39 ms per token
# End-to-end sequence generation
# - with use_cache=True
#         37.04 ms per token
# - with use_cache=False
#         90.68 ms per token
# Compiling model...
# Compiled results:
# - with use_cache=True
#         18.66 ms per token
# - with use_cache=False
#         67.66 ms per token

# (Compiled) End-to-end sequence generation
# - with use_cache=True
#         20.61 ms per token
# - with use_cache=False
#         71.45 ms per token


parser = argparse.ArgumentParser(
    description="Script to benchmark inference time per token on a LLaMA model"
)
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--architecture",
    type=str,
    default="llama",
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default="7b",
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--seq_len",
    type=int,
    default=512,
    help="Sequence length of mock input",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    help="Batch size of mock input",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=256,
    help="Max number of tokens to generate",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for compilation",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set seeds and torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument(
    "--skip_correctness_check",
    action="store_true",
    help="Do not test correctness of outputs vs just timing",
)
parser.add_argument(
    "--skip_eager_runs", action="store_true", help="Do not run the eager benchmarks"
)
parser.add_argument(
    "--skip_compile_runs",
    action="store_true",
    help="Do not run the compiled benchmarks",
)
parser.add_argument(
    "--skip_kvcache_runs",
    action="store_true",
    help="Do not run the kv-cache benchmarks",
)
parser.add_argument(
    "--skip_nokvcache_runs",
    action="store_true",
    help="Do not run the no kv-cache benchmarks",
)
parser.add_argument(
    "--skip_single_token_runs",
    action="store_true",
    help="Do not run the single token benchmarks",
)
parser.add_argument(
    "--skip_e2e_runs", action="store_true", help="Do not run the e2e benchmarks"
)
parser.add_argument(
    "--unfuse_weights",
    action="store_true",
    help="If set to True, this will unfuse any fused weight modules that support the unfuse_weights method",
)
parser.add_argument(
    "--profile_memory",
    action="store_true",
    help="Profile peak memory usage for a single forward pass (current --seq_len) and print result."
)
parser.add_argument(
    "--profile_throughput",
    action="store_true",
    help="Profile throughput for multiple concurrent generation requests."
)
parser.add_argument(
    "--num_requests",
    type=int,
    default=4,
    help="Number of concurrent requests for throughput profiling."
)
parser.add_argument(
    "--output_csv",
    type=str,
    default=None,
    help="Path to output CSV file for benchmark results."
)
parser.add_argument(
    "--run_runtime_vs_seqlen_sweep",
    action="store_true",
    help="Run runtime (fwd+bwd) vs sequence length sweep for paged and non-paged, save to CSV, then exit."
)
parser.add_argument(
    "--run_memory_vs_seqlen_sweep",
    action="store_true",
    help="Run memory usage vs sequence length sweep for paged and non-paged, save to CSV, then exit."
)

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)  # pytorch random seed
    np.random.seed(SEED)  # numpy random seed
    torch.use_deterministic_algorithms(True)

if world_size > 1:
    dist.init_process_group()
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

print("loading model")
model = models.get_model(args.architecture, args.variant, device_type=args.device_type)

if args.unfuse_weights:
    print("unfusing weights")
    model = fusion.apply_unfuse_weights(model)

tokenizer = tokenizers.get_tokenizer(args.tokenizer)

model.eval()
torch.set_grad_enabled(False)
print(f"loading complete on rank {local_rank}")

SEQ_LEN = args.seq_len
BATCH_SIZE = args.batch_size
MAX_NEW_TOKENS = args.max_new_tokens

ids = torch.randint(
    tokenizer.vocab_size(), (BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.long
)

# This first forward call generates the cache for use in cases where
# `use_cache=True`.
#
# For performance purposes, this call can be considered equivalent to
# `use_cache=False`.
#
# The actual performance of generation with `use_cache=True` would be the cost
# of the first token without cache, plus the cost of all subsequent tokens with
# cache. I.e. the amortized per-token cost would depend on the number of tokens
# generated.
logits, cache = model.forward(ids, use_cache=True)
logits = logits[:, -1, :]
next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()
next_input = torch.cat((ids, next_val), dim=-1)

# not still needed
del logits

expected, _ = model.forward(
    next_val, past_key_value_states=cache, use_cache=True, only_last_token=True
)
expected = torch.argmax(expected, dim=-1)

expected2 = model.forward(next_input, only_last_token=True)
expected2 = torch.argmax(expected2, dim=-1)

torch.testing.assert_close(expected, expected2)

repeat = 3


# The function we're measuring, with or without caching.
#
# In a realistic generate function, the sequence length would grow with each
# subsequent token, and so the average cost would be from a variety of sequence
# lengths.
# We capture the time to generate a single token from a given sequence length
# and batch size. This means we're measuring the cost of the forward pass
# in isolation in a way that's easier to compare, and avoids including the cost
# of the concatenation operation.
def one_token(model, use_cache):
    if use_cache:
        actual, _ = model.forward(
            next_val, past_key_value_states=cache, use_cache=True, only_last_token=True
        )
    else:
        actual = model.forward(next_input, only_last_token=True)
    actual = torch.argmax(actual, dim=-1)
    if local_rank == 0 and not args.skip_correctness_check:
        torch.testing.assert_close(actual, expected)
    else:
        if args.device_type == "cuda":
            torch.cuda.synchronize()


def end_to_end(model, use_cache, expected=None):
    result = generation.generate(
        model,
        ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=use_cache,
        contiguous_cache=args.compile_mode == "reduce-overhead"
        and isinstance(
            model, OptimizedModule
        ),  # this is needed for reduce-overhead to work correctly for now
    )
    if local_rank == 0:
        assert result.size()[-1] == SEQ_LEN + MAX_NEW_TOKENS, (
            f"{result.size()}, {SEQ_LEN}, {MAX_NEW_TOKENS}"
        )
    if expected is not None and not args.skip_correctness_check:
        torch.testing.assert_close(result, expected)
    else:
        if args.device_type == "cuda":
            torch.cuda.synchronize()
    return result


e2e_expected_cache = end_to_end(model, True)
e2e_expected_nocache = end_to_end(model, True)


def log_result(result):
    if local_rank == 0:
        median = statistics.median(result)
        per_token = median / MAX_NEW_TOKENS
        ms = per_token * 1000
        print(f"\t{ms:0.2f} ms per token")


def bench_one(use_cache):
    print0(f"- with use_cache={use_cache}")
    log_result(
        timeit.repeat(
            lambda: one_token(model, use_cache), number=MAX_NEW_TOKENS, repeat=repeat
        )
    )


def bench_end_to_end(use_cache, expected):
    print0(f"- with use_cache={use_cache}")
    result = timeit.repeat(
        lambda: end_to_end(model, use_cache, expected), number=1, repeat=repeat
    )
    log_result(result)


print0(
    f"Results for batch size {BATCH_SIZE}, sequence length {SEQ_LEN}, new tokens generated {MAX_NEW_TOKENS}"
)
if not args.skip_eager_runs:
    print0("Uncompiled results:")
    print0("==========")
    if not args.skip_single_token_runs:
        print0("Single token generation")
        if not args.skip_kvcache_runs:
            bench_one(True)
        if not args.skip_nokvcache_runs:
            bench_one(False)

    if not args.skip_e2e_runs:
        print0("End-to-end sequence generation")
        if not args.skip_kvcache_runs:
            bench_end_to_end(True, e2e_expected_cache)
        if not args.skip_nokvcache_runs:
            bench_end_to_end(False, e2e_expected_nocache)

if not args.skip_compile_runs:
    print0("Compiling model...")

    # This is to prevent a bug in PT 2.1 that has been fixed in PT 2.2 nightlies
    torch._inductor.config.joint_graph_constant_folding = False
    # with mode='reduce-overhead' we see better performance but on multi-GPU models
    # hit an error on the end-to-end test below when run after other tests (if it's
    # run first it works, confirming a memory leak):
    # `RuntimeError: Expected curr_block->ptr == block_state.ptr to be true, but got false.`
    model = torch.compile(model, dynamic=True, mode=args.compile_mode)

    print0()
    print0("Compiled results:")
    print0("==========")

    if not args.skip_single_token_runs:
        # Warmup. Especially with torch.compile, first inference pass can be slow.
        print(f"Warming up the compiled model for single token in rank {local_rank}")
        # Activate dynamo logs to ensure some output during compilation
        torch._logging.set_logs(dynamo=logging.INFO)
        if not args.skip_kvcache_runs:
            one_token(model, True)
        if not args.skip_nokvcache_runs:
            one_token(model, False)
        print(f"Model has warmed up in rank {local_rank}")

        # These get much better results with mode='reduce-overhead' but can lead to
        # some memory issues
        print0("(Compiled) Single token generation")
        if not args.skip_kvcache_runs:
            bench_one(True)
        if not args.skip_nokvcache_runs:
            bench_one(False)

    if not args.skip_e2e_runs:
        print0()
        print(f"Warming up the compiled model e2e in rank {local_rank}")
        if not args.skip_kvcache_runs:
            end_to_end(model, True, e2e_expected_cache)
        if not args.skip_nokvcache_runs:
            end_to_end(model, False, e2e_expected_nocache)
        print(f"Model has warmed up e2e in rank {local_rank}")

        print0("(Compiled) End-to-end sequence generation")
        if not args.skip_kvcache_runs:
            bench_end_to_end(True, e2e_expected_cache)
        if not args.skip_nokvcache_runs:
            bench_end_to_end(False, e2e_expected_nocache)

def profile_memory(model, tokenizer, device, batch_size, seq_len):
    """Print peak GPU memory usage (GB) for a single forward pass at the given sequence length."""
    # Ensure we are running on a CUDA device, since memory profiling is only meaningful on GPU.
    if not (hasattr(device, 'type') and device.type == 'cuda'):
        raise RuntimeError('profile_memory requires a CUDA device.')
    # Generate random input IDs to simulate a real input batch for the model.
    ids = torch.randint(
        tokenizer.vocab_size(), (batch_size, seq_len), device=device, dtype=torch.long
    )
    # Clear any cached memory to ensure a clean measurement of peak usage.
    torch.cuda.empty_cache()
    # Reset PyTorch's peak memory statistics so our measurement is accurate for this run.
    torch.cuda.reset_peak_memory_stats()
    # Disable gradient tracking for inference-only memory measurement (saves memory and compute).
    with torch.no_grad():
        _ = model.forward(ids, use_cache=True)
    # Get the peak memory allocated during the forward pass, convert to GB for readability.
    peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
    # Print the result in a clear, parseable format for downstream analysis or logging.
    print(f"Peak memory usage (GB): {peak_mem:.4f}")

# After all setup (argument parsing, model/tokenizer/device setup), add this at the top level:
if args.profile_memory:
    profile_memory(model, tokenizer, device, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

def profile_throughput(model, tokenizer, device, batch_size, seq_len, num_requests):
    """Profile throughput and latency for multiple concurrent generation requests (true concurrency with threads)."""
    if not (hasattr(device, 'type') and device.type == 'cuda'):
        raise RuntimeError('profile_throughput requires a CUDA device.')
    # Generate random input IDs for each request
    requests = [
        torch.randint(
            tokenizer.vocab_size(), (batch_size, seq_len), device=device, dtype=torch.long
        )
        for _ in range(num_requests)
    ]
    # Clear any cached memory to ensure a clean measurement of peak usage.
    torch.cuda.empty_cache()
    # Reset PyTorch's peak memory statistics so our measurement is accurate for this run.
    torch.cuda.reset_peak_memory_stats()
    # Time the concurrent forward passes using threads to simulate real-world multi-request serving
    import timeit
    results = [None] * num_requests
    def run_forward(i):
        with torch.no_grad():
            results[i] = model.forward(requests[i], use_cache=True)
    threads = [threading.Thread(target=run_forward, args=(i,)) for i in range(num_requests)]
    start_time = timeit.default_timer()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end_time = timeit.default_timer()
    # Calculate metrics
    total_time = end_time - start_time
    total_tokens = num_requests * batch_size * seq_len
    throughput = total_tokens / total_time if total_time > 0 else float('inf')
    avg_latency = (total_time / total_tokens) * 1000 if total_tokens > 0 else 0  # ms per token
    peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
    # Print results in a clear, parseable format
    print(f"Throughput (tokens/sec): {throughput:.2f}")
    print(f"Average latency (ms/token): {avg_latency:.2f}")
    print(f"Peak memory usage (GB): {peak_mem:.4f}")

# Top-level call for throughput profiling
if args.profile_throughput:
    profile_throughput(model, tokenizer, device, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_requests=args.num_requests)

def benchmark_throughput_vs_seqlen(model, tokenizer, device, batch_size, max_seq_len=4096, step=512):
    """Benchmark throughput at different sequence lengths with max sustainable batch size."""
    seq_lengths = range(512, max_seq_len + 1, step)
    results = []
    
    for seq_len in seq_lengths:
        # Generate input
        ids = torch.randint(tokenizer.vocab_size(), (batch_size, seq_len), device=device, dtype=torch.long)
        
        # Warmup
        for _ in range(3):
            model.forward(ids, use_cache=True)
        
        # Measure throughput
        start_time = time.time()
        num_iterations = 10
        for _ in range(num_iterations):
            model.forward(ids, use_cache=True)
        end_time = time.time()
        
        # Calculate tokens per second
        total_tokens = batch_size * seq_len * num_iterations
        tokens_per_second = total_tokens / (end_time - start_time)
        
        results.append({
            'seq_len': seq_len,
            'throughput': tokens_per_second,
            'batch_size': batch_size
        })
        
    return results

def benchmark_throughput_vs_batchsize(model, tokenizer, device, seq_len, max_batch_size=32, step=2):
    """Benchmark throughput at different batch sizes with fixed sequence length."""
    batch_sizes = range(1, max_batch_size + 1, step)
    results = []
    
    for batch_size in batch_sizes:
        # Generate input
        ids = torch.randint(tokenizer.vocab_size(), (batch_size, seq_len), device=device, dtype=torch.long)
        
        # Warmup
        for _ in range(3):
            model.forward(ids, use_cache=True)
        
        # Measure throughput
        start_time = time.time()
        num_iterations = 10
        for _ in range(num_iterations):
            model.forward(ids, use_cache=True)
        end_time = time.time()
        
        # Calculate tokens per second
        total_tokens = batch_size * seq_len * num_iterations
        tokens_per_second = total_tokens / (end_time - start_time)
        
        results.append({
            'batch_size': batch_size,
            'throughput': tokens_per_second,
            'seq_len': seq_len
        })
        
    return results

def benchmark_latency(model, tokenizer, device, batch_size, seq_len, num_tokens=100):
    """Analyze latency per token and per sequence."""
    # Generate input
    ids = torch.randint(tokenizer.vocab_size(), (batch_size, seq_len), device=device, dtype=torch.long)
    
    # Warmup
    for _ in range(3):
        model.forward(ids, use_cache=True)
    
    # Measure token generation latency
    token_latencies = []
    sequence_latencies = []
    
    for _ in range(num_tokens):
        # Measure per-token latency
        start_time = time.time()
        logits, cache = model.forward(ids, use_cache=True)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        end_time = time.time()
        token_latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Update input for next token
        ids = torch.cat([ids, next_token.unsqueeze(-1)], dim=-1)
        
        # Measure full sequence latency every 10 tokens
        if len(token_latencies) % 10 == 0:
            seq_start = time.time()
            model.forward(ids, use_cache=True)
            seq_end = time.time()
            sequence_latencies.append((seq_end - seq_start) * 1000)  # Convert to ms
    
    return {
        'avg_token_latency_ms': statistics.mean(token_latencies),
        'p95_token_latency_ms': np.percentile(token_latencies, 95),
        'avg_sequence_latency_ms': statistics.mean(sequence_latencies),
        'p95_sequence_latency_ms': np.percentile(sequence_latencies, 95)
    }

def run_benchmarks(model, tokenizer, device, args):
    """Run all benchmark tests."""
    results = {}
    
    # Throughput vs Sequence Length
    print("\nRunning Throughput vs Sequence Length benchmark...")
    results['throughput_vs_seqlen'] = benchmark_throughput_vs_seqlen(
        model, tokenizer, device, 
        batch_size=args.batch_size,
        max_seq_len=args.seq_len
    )
    
    # Throughput vs Batch Size
    print("\nRunning Throughput vs Batch Size benchmark...")
    results['throughput_vs_batchsize'] = benchmark_throughput_vs_batchsize(
        model, tokenizer, device,
        seq_len=args.seq_len,
        max_batch_size=args.batch_size * 4
    )
    
    # Latency Analysis
    print("\nRunning Latency Analysis benchmark...")
    results['latency'] = benchmark_latency(
        model, tokenizer, device,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )
    
    return results

def runtime_vs_seqlen_sweep(tokenizer, device, paged, output_csv, batch_size=1, min_seq=128, max_seq=8192, step=128):
    """Run runtime (fwd+bwd) vs sequence length for paged/non-paged, save to CSV."""
    import torch.nn as nn
    import torch.optim as optim
    results = []
    os.environ["FMS_ATTENTION_ALGO"] = "paged" if paged else ""
    model = models.get_model(args.architecture, args.variant, device_type=args.device_type)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for seq_len in range(min_seq, max_seq+1, step):
        ids = torch.randint(tokenizer.vocab_size(), (batch_size, seq_len), device=device, dtype=torch.long)
        targets = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
        # Warmup
        for _ in range(2):
            logits = model(ids)[0] if isinstance(model(ids), tuple) else model(ids)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.zero_grad()
        # Timed run
        torch.cuda.synchronize()
        start = time.time()
        logits = model(ids)[0] if isinstance(model(ids), tuple) else model(ids)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        torch.cuda.synchronize()
        end = time.time()
        runtime_ms = (end - start) * 1000
        results.append({"seq_len": seq_len, "runtime_ms": runtime_ms, "paged": paged})
    if output_csv:
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seq_len", "runtime_ms", "paged"])
            if f.tell() == 0:
                writer.writeheader()
            for row in results:
                writer.writerow(row)
    return results

def memory_vs_seqlen_sweep(tokenizer, device, paged, output_csv, batch_size=1, min_seq=256, max_seq=65536, step=1024):
    """Run memory usage vs sequence length for paged/non-paged, save to CSV."""
    results = []
    os.environ["FMS_ATTENTION_ALGO"] = "paged" if paged else ""
    model = models.get_model(args.architecture, args.variant, device_type=args.device_type)
    model.eval()
    for seq_len in range(min_seq, max_seq+1, step):
        ids = torch.randint(tokenizer.vocab_size(), (batch_size, seq_len), device=device, dtype=torch.long)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model.forward(ids, use_cache=True)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
        results.append({"seq_len": seq_len, "memory_gb": peak_mem, "paged": paged})
    if output_csv:
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seq_len", "memory_gb", "paged"])
            if f.tell() == 0:
                writer.writeheader()
            for row in results:
                writer.writerow(row)
    return results

if __name__ == "__main__":
    # Run sweep and exit if requested
    if args.run_runtime_vs_seqlen_sweep:
        if args.output_csv and os.path.exists(args.output_csv):
            os.remove(args.output_csv)
        print("Running runtime vs sequence length (non-paged)...")
        runtime_vs_seqlen_sweep(tokenizer, device, paged=False, output_csv=args.output_csv)
        print("Running runtime vs sequence length (paged)...")
        runtime_vs_seqlen_sweep(tokenizer, device, paged=True, output_csv=args.output_csv)
        print(f"Results written to {args.output_csv}")
        exit(0)
    if args.run_memory_vs_seqlen_sweep:
        if args.output_csv and os.path.exists(args.output_csv):
            os.remove(args.output_csv)
        print("Running memory vs sequence length (non-paged)...")
        memory_vs_seqlen_sweep(tokenizer, device, paged=False, output_csv=args.output_csv)
        print("Running memory vs sequence length (paged)...")
        memory_vs_seqlen_sweep(tokenizer, device, paged=True, output_csv=args.output_csv)
        print(f"Results written to {args.output_csv}")
        exit(0)
    
    if args.profile_memory:
        profile_memory(model, tokenizer, device, BATCH_SIZE, SEQ_LEN)
    elif args.profile_throughput:
        profile_throughput(model, tokenizer, device, BATCH_SIZE, SEQ_LEN, args.num_requests)
    else:
        # Run all benchmarks
        results = run_benchmarks(model, tokenizer, device, args)
        
        # Print results
        print("\nBenchmark Results:")
        print("\nThroughput vs Sequence Length:")
        for r in results['throughput_vs_seqlen']:
            print(f"Seq Len: {r['seq_len']}, Throughput: {r['throughput']:.2f} tokens/sec")
            
        print("\nThroughput vs Batch Size:")
        for r in results['throughput_vs_batchsize']:
            print(f"Batch Size: {r['batch_size']}, Throughput: {r['throughput']:.2f} tokens/sec")
            
        print("\nLatency Analysis:")
        print(f"Average Token Latency: {results['latency']['avg_token_latency_ms']:.2f} ms")
        print(f"P95 Token Latency: {results['latency']['p95_token_latency_ms']:.2f} ms")
        print(f"Average Sequence Latency: {results['latency']['avg_sequence_latency_ms']:.2f} ms")
        print(f"P95 Sequence Latency: {results['latency']['p95_sequence_latency_ms']:.2f} ms")
