from pathlib import Path
import sys
import os
import time
from random import randint, seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    block_manager = llm.scheduler.block_manager
    block_manager.reset_stats()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    wall_time = time.perf_counter() - wall_start
    cpu_time = time.process_time() - cpu_start
    block_stats = block_manager.get_stats()
    block_time = block_stats["total_time"]
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / wall_time
    print(f"Total: {total_tokens}tok, Time: {wall_time:.2f}s, Throughput: {throughput:.2f}tok/s")
    print(f"CPU process time: {cpu_time:.4f}s")
    print(f"BlockManager total time: {block_time:.4f}s ({block_time / wall_time * 100:.2f}% of wall time)")
    print("BlockManager breakdown:")
    for name in ("can_allocate", "get_token_layout", "allocate", "deallocate", "can_append", "may_append"):
        print(
            f"  {name}: {block_stats[f'{name}_time']:.4f}s, "
            f"calls={block_stats[f'{name}_calls']}"
        )


if __name__ == "__main__":
    main()
