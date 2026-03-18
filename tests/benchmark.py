import argparse
import gc
import os
from pathlib import Path
import statistics
import sys
import time
from random import Random


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
import torch


class NoPrefixCacheBlockManager(BlockManager):

    def get_token_layout(self, seq):
        self.stats["get_token_layout_calls"] += 1
        start = time.perf_counter()
        try:
            assert not seq.block_table
            return 0, 0, seq.num_tokens
        finally:
            self.stats["get_token_layout_time"] += time.perf_counter() - start

    def allocate(self, seq):
        self.stats["allocate_calls"] += 1
        start = time.perf_counter()
        try:
            assert not seq.block_table
            for i in range(0, seq.num_new_tokens, self.block_size):
                token_ids = seq[i: min(i + self.block_size, seq.num_new_tokens)]
                block = self._allocate_new_block()
                block.update(token_ids)
                seq.block_table.append(block.block_id)
        finally:
            self.stats["allocate_time"] += time.perf_counter() - start

    def may_append(self, seq):
        self.stats["may_append_calls"] += 1
        start = time.perf_counter()
        try:
            for i in range(
                seq.num_cached_blocks * self.block_size,
                seq.num_cached_tokens + seq.num_new_tokens,
                self.block_size,
            ):
                token_ids = seq[i: min(i + self.block_size, seq.num_cached_tokens + seq.num_new_tokens)]
                block_index = i // self.block_size
                current_block_id = seq.block_table[block_index] if block_index < len(seq.block_table) else -1
                if current_block_id == -1:
                    current_block = self._allocate_new_block()
                    seq.block_table.append(current_block.block_id)
                else:
                    current_block = self.blocks[current_block_id]
                    assert current_block.node is None
                current_block.update(token_ids)
        finally:
            self.stats["may_append_time"] += time.perf_counter() - start


def build_full_miss_workload(
    num_seqs: int,
    max_input_len: int,
    max_output_len: int,
    seed: int,
) -> tuple[list[list[int]], list[SamplingParams]]:
    rng = Random(seed)
    prompts = []
    sampling_params = []
    for req_id in range(num_seqs):
        prompt_len = rng.randint(100, max_input_len)
        prompt = [rng.randint(0, 10000) for _ in range(prompt_len)]
        # Make the first block unique so prefix matching always misses at root.
        prompt[0] = req_id
        if prompt_len > 1:
            prompt[1] = req_id + num_seqs
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                temperature=0.6,
                ignore_eos=True,
                max_tokens=rng.randint(100, max_output_len),
            )
        )
    return prompts, sampling_params


def build_high_hit_workload(
    num_seqs: int,
    max_input_len: int,
    max_output_len: int,
    seed: int,
    shared_prefix_len: int,
    unique_suffix_len: int,
) -> tuple[list[list[int]], list[SamplingParams]]:
    rng = Random(seed)
    prefix_len = min(shared_prefix_len, max_input_len)
    suffix_len = min(unique_suffix_len, max(max_input_len - prefix_len, 0))
    if prefix_len + suffix_len == 0:
        prefix_len = min(1, max_input_len)
    shared_prefix = [rng.randint(0, 10000) for _ in range(prefix_len)]
    prompts = []
    sampling_params = []
    for req_id in range(num_seqs):
        prompt = list(shared_prefix)
        for _ in range(suffix_len):
            prompt.append(rng.randint(0, 10000))
        if suffix_len > 0:
            # Keep the tail request-specific, but preserve a large shared prefix.
            prompt[-1] = req_id
        if not prompt:
            prompt = [req_id]
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                temperature=0.6,
                ignore_eos=True,
                max_tokens=rng.randint(100, max_output_len),
            )
        )
    return prompts, sampling_params


def build_workload(args, run_index: int):
    seed = args.seed + run_index
    if args.workload == "full-miss":
        return build_full_miss_workload(
            num_seqs=args.num_seqs,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            seed=seed,
        )
    if args.workload == "high-hit":
        return build_high_hit_workload(
            num_seqs=args.num_seqs,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            seed=seed,
            shared_prefix_len=args.shared_prefix_len,
            unique_suffix_len=args.unique_suffix_len,
        )
    raise ValueError(f"Unsupported workload: {args.workload}")


def build_engine(args, enable_prefix_cache: bool) -> LLM:
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        chunked_prefill=args.chunked_prefill,
    )
    if not enable_prefix_cache:
        original = llm.scheduler.block_manager
        llm.scheduler.block_manager = NoPrefixCacheBlockManager(
            num_blocks=len(original.blocks),
            block_size=original.block_size,
        )
    return llm


def warmup(llm: LLM):
    llm.generate(["Benchmark warmup"], SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)


def run_once(args, mode_name: str, enable_prefix_cache: bool, prompts, sampling_params):
    llm = build_engine(args, enable_prefix_cache=enable_prefix_cache)
    try:
        warmup(llm)
        block_manager = llm.scheduler.block_manager
        block_manager.reset_stats()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        wall_time = time.perf_counter() - wall_start
        cpu_time = time.process_time() - cpu_start
        block_stats = block_manager.get_stats()
        total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_output_tokens / wall_time
        return {
            "mode": mode_name,
            "wall_time": wall_time,
            "cpu_time": cpu_time,
            "throughput": throughput,
            "total_output_tokens": total_output_tokens,
            "block_stats": block_stats,
        }
    finally:
        llm.exit()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_result(result):
    block_total = result["block_stats"]["total_time"]
    print(f"[{result['mode']}]")
    print(
        f"  output_tokens={result['total_output_tokens']}  "
        f"wall_time={result['wall_time']:.4f}s  "
        f"cpu_time={result['cpu_time']:.4f}s  "
        f"throughput={result['throughput']:.2f} tok/s"
    )
    print(
        f"  block_manager_total={block_total:.4f}s  "
        f"share_of_wall={block_total / result['wall_time'] * 100:.2f}%"
    )
    print("  block_manager_breakdown:")
    for name in ("can_allocate", "get_token_layout", "allocate", "deallocate", "can_append", "may_append"):
        print(
            f"    {name:<16} "
            f"time={result['block_stats'][f'{name}_time']:.4f}s "
            f"calls={result['block_stats'][f'{name}_calls']}"
        )


def comparison_metrics(baseline, radix):
    return {
        "wall_time_overhead": (radix["wall_time"] / baseline["wall_time"] - 1.0) * 100,
        "cpu_time_overhead": (radix["cpu_time"] / baseline["cpu_time"] - 1.0) * 100,
        "throughput_delta": (radix["throughput"] / baseline["throughput"] - 1.0) * 100,
        "block_manager_extra_time": radix["block_stats"]["total_time"] - baseline["block_stats"]["total_time"],
    }


def print_comparison(metrics):
    print("[comparison: radix_prefix_cache vs no_prefix_cache]")
    print(f"  wall_time_overhead={metrics['wall_time_overhead']:.2f}%")
    print(f"  cpu_time_overhead={metrics['cpu_time_overhead']:.2f}%")
    print(f"  throughput_delta={metrics['throughput_delta']:.2f}%")
    print(f"  block_manager_extra_time={metrics['block_manager_extra_time']:.4f}s")


def summarize_mode(mode_name: str, results: list[dict]):
    wall_times = [r["wall_time"] for r in results]
    cpu_times = [r["cpu_time"] for r in results]
    throughputs = [r["throughput"] for r in results]
    block_times = [r["block_stats"]["total_time"] for r in results]
    print(f"[summary: {mode_name}]")
    print(
        f"  wall_time_mean={statistics.mean(wall_times):.4f}s  "
        f"wall_time_std={statistics.pstdev(wall_times):.4f}s"
    )
    print(
        f"  cpu_time_mean={statistics.mean(cpu_times):.4f}s  "
        f"cpu_time_std={statistics.pstdev(cpu_times):.4f}s"
    )
    print(
        f"  throughput_mean={statistics.mean(throughputs):.2f} tok/s  "
        f"throughput_std={statistics.pstdev(throughputs):.2f}"
    )
    print(
        f"  block_manager_mean={statistics.mean(block_times):.4f}s  "
        f"block_manager_std={statistics.pstdev(block_times):.4f}s"
    )


def summarize_comparisons(comparisons: list[dict]):
    print("[summary: radix_prefix_cache vs no_prefix_cache]")
    for key in ("wall_time_overhead", "cpu_time_overhead", "throughput_delta", "block_manager_extra_time"):
        values = [item[key] for item in comparisons]
        if key.endswith("_time") or "extra_time" in key:
            print(f"  {key}_mean={statistics.mean(values):.4f}s  {key}_std={statistics.pstdev(values):.4f}s")
        else:
            print(f"  {key}_mean={statistics.mean(values):.2f}%  {key}_std={statistics.pstdev(values):.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark full-miss prefix-cache overhead for nano-vLLM.",
    )
    parser.add_argument("--model", type=str, required=True, help="Local model path.")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--workload", type=str, default="full-miss", choices=("full-miss", "high-hit"))
    parser.add_argument("--shared-prefix-len", type=int, default=768)
    parser.add_argument("--unique-suffix-len", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--chunked-prefill", action="store_true", default=False)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    args.model = os.path.expanduser(args.model)

    print("Running prefix-cache benchmark with identical workloads for both modes.")
    print(
        f"workload={args.workload}, num_seqs={args.num_seqs}, "
        f"max_input_len={args.max_input_len}, max_output_len={args.max_output_len}, "
        f"chunked_prefill={args.chunked_prefill}, num_runs={args.num_runs}"
    )
    if args.workload == "high-hit":
        print(
            f"shared_prefix_len={args.shared_prefix_len}, "
            f"unique_suffix_len={args.unique_suffix_len}"
        )

    baseline_results = []
    radix_results = []
    comparisons = []
    for run_index in range(args.num_runs):
        print(f"\n=== Run {run_index + 1}/{args.num_runs} ===")
        prompts, sampling_params = build_workload(args, run_index)
        baseline = run_once(args, "no_prefix_cache", False, prompts, sampling_params)
        radix = run_once(args, "radix_prefix_cache", True, prompts, sampling_params)
        baseline_results.append(baseline)
        radix_results.append(radix)
        metrics = comparison_metrics(baseline, radix)
        comparisons.append(metrics)
        print_result(baseline)
        print_result(radix)
        print_comparison(metrics)

    print()
    summarize_mode("no_prefix_cache", baseline_results)
    summarize_mode("radix_prefix_cache", radix_results)
    summarize_comparisons(comparisons)


if __name__ == "__main__":
    main()
