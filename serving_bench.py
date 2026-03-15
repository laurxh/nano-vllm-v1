import os
import time
import numpy as np
import argparse
from random import randint, seed
from tqdm.auto import tqdm
from nanovllm import LLM, SamplingParams


# python /data/slwang/nano-vllm-online-running-first-schedule/serving_bench.py --model /data/hfhub/Qwen3/Qwen3-14B/ --request-rate 10 --num-requests 1024 --tensor-parallel-size 1 --max-num-batched-tokens 1024 --max-num-seqs 1024 --random-input-len 128 --random-output-len 100 --chunked-prefill --enforce-eager

# --- Seed for reproducibility ---
seed(100)
np.random.seed(100)

class RequestMetrics:
    """Stores metrics for a single request."""
    def __init__(self, request_id, input_len):
        self.request_id = request_id
        self.input_len = input_len
        self.submission_time = -1
        self.first_token_time = -1
        self.completion_time = -1
        self.output_len = -1

    # def record_submission(self):
    #     self.submission_time = time.perf_counter()

    def record_first_token(self):
        if self.first_token_time == -1:
            self.first_token_time = time.perf_counter()

    def record_completion(self, output_ids):
        self.completion_time = time.perf_counter()
        self.output_len = len(output_ids)

    @property
    def ttft(self):
        return self.first_token_time - self.submission_time

    @property
    def tpot(self):
        if self.output_len > 1:
            return (self.completion_time - self.first_token_time) / (self.output_len - 1)
        return float('nan')

    @property
    def latency(self):
        return self.completion_time - self.submission_time


def percentile_ms(values, q):
    if not values:
        return float('nan')
    return float(np.percentile(values, q)) * 1000


def warm_up(engine, args):
    # --- Generate random prompts ---
    # prompts = [[randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))] for _ in range(NUM_REQUESTS)]
    prompts = [[randint(0, 10000) for _ in range(args.random_input_len)] for _ in range(50)]
    # sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, MAX_OUTPUT_LEN)) for _ in range(NUM_REQUESTS)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.random_output_len) for _ in range(args.num_requests)]
    outputs = engine.generate(prompts, sampling_params)


def main():
    """Main function to run the serving benchmark."""
    parser = argparse.ArgumentParser(description="Serving benchmark for nano-vllm.")
    parser.add_argument('--model', type=str, required=True, help="Model name or path (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--num-requests", type=int, default=256, help="Number of requests to process.")
    parser.add_argument("--request-rate", type=int, default=8, help="Request rate (requests per second).")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--random-input-len", type=int, default=128)
    parser.add_argument("--random-output-len", type=int, default=128)
    parser.add_argument('--chunked-prefill', action='store_true', default=False, help="Enable chunked prefill mode.")
    parser.add_argument('--enforce-eager', action='store_true', default=False, help="Enforce eager execution mode.")
    args = parser.parse_args()

    print(f"\n--- Running benchmark with --num-requests {args.num_requests} --request-rate {args.request_rate} ---")
    llm = LLM(
        args.model, 
        enforce_eager=args.enforce_eager, 
        max_model_len=40960, 
        max_num_batched_tokens=args.max_num_batched_tokens, 
        max_num_seqs=args.max_num_seqs, 
        tensor_parallel_size=args.tensor_parallel_size,
        chunked_prefill=args.chunked_prefill,
    )
    engine = llm
    
    warm_up(engine, args)

    # --- Generate random prompts ---
    # prompts = [[randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))] for _ in range(NUM_REQUESTS)]
    prompts = [[randint(0, 10000) for _ in range(args.random_input_len)] for _ in range(args.num_requests)]
    # sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, MAX_OUTPUT_LEN)) for _ in range(NUM_REQUESTS)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.random_output_len) for _ in range(args.num_requests)]

    # --- Generate request arrival times ---
    request_intervals = np.random.exponential(1.0 / args.request_rate, args.num_requests)
    arrival_times = np.cumsum(request_intervals)
    print(arrival_times)

    # --- Benchmark loop ---
    metrics = {}
    requests_sent = 0
    start_time = time.perf_counter()
    completed_latencies = []

    with tqdm(total=args.num_requests, desc="Processing Requests") as pbar:
        while requests_sent < args.num_requests or not engine.is_finished():
            # --- Send new requests ---
            current_time = time.perf_counter()
            while requests_sent < args.num_requests and current_time - start_time >= arrival_times[requests_sent]:
                prompt = prompts[requests_sent]
                sp = sampling_params[requests_sent]
                
                engine.add_request(prompt, sp)
                
                new_seq = engine.scheduler.waiting[-1]
                seq_id = new_seq.seq_id
                req_metrics = RequestMetrics(seq_id, len(prompt))
                req_metrics.submission_time = start_time + arrival_times[requests_sent]
                metrics[seq_id] = req_metrics
                
                requests_sent += 1

            # --- Engine step ---
            if not engine.is_finished():
                finished_outputs, _ = engine.step()

                # Record first token time for all processed sequences
                all_processed_seqs = list(engine.scheduler.running)
                for seq in all_processed_seqs:
                    if seq.seq_id in metrics and seq.num_cached_tokens == seq.num_prompt_tokens:
                        metrics[seq.seq_id].record_first_token()

                for seq_id, output_ids in finished_outputs:
                    if seq_id in metrics:
                        metrics[seq_id].record_first_token() # Ensure first token time is recorded
                        metrics[seq_id].record_completion(output_ids)
                        
                        completed_latencies.append(metrics[seq_id].latency)
                        avg_latency = np.mean(completed_latencies)
                        pbar.set_postfix({"Avg Latency": f"{avg_latency:.2f}s"})
                        pbar.update(1)
            else:
                # If no requests are running or waiting, sleep briefly
                time.sleep(0.01)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # --- Calculate and print metrics ---
    total_input_tokens = sum(m.input_len for m in metrics.values())
    total_output_tokens = sum(m.output_len for m in metrics.values() if m.output_len != -1)
    
    avg_ttft = np.mean([m.ttft for m in metrics.values() if m.first_token_time != -1])
    avg_tpot = np.mean([m.tpot for m in metrics.values() if not np.isnan(m.tpot)])
    avg_latency = np.mean([m.latency for m in metrics.values() if m.completion_time != -1])
    ttfts = [m.ttft for m in metrics.values() if m.first_token_time != -1]
    latencies = [m.latency for m in metrics.values() if m.completion_time != -1]
    throughput = (total_input_tokens + total_output_tokens) / total_time

    print("--- Benchmark Results ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests sent: {requests_sent}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Average TTFT: {avg_ttft * 1000:.2f} ms")
    print(f"P50 TTFT: {percentile_ms(ttfts, 50):.2f} ms")
    print(f"P95 TTFT: {percentile_ms(ttfts, 95):.2f} ms")
    print(f"P99 TTFT: {percentile_ms(ttfts, 99):.2f} ms")
    print(f"Average TPOT: {avg_tpot * 1000:.2f} ms")
    print(f"Average latency: {avg_latency:.2f} s")
    print(f"P50 latency: {np.percentile(latencies, 50):.2f} s")
    print(f"P95 latency: {np.percentile(latencies, 95):.2f} s")
    print(f"P99 latency: {np.percentile(latencies, 99):.2f} s")
    print("-------------------------\n")

if __name__ == "__main__":
    main()
"""
chunked prefill
Total time: 25.89s
Requests sent: 500
Throughput: 22247.67 tokens/s
Average TTFT: 56.48 ms
P50 TTFT: 51.10 ms
P95 TTFT: 93.67 ms
P99 TTFT: 111.61 ms
Average TPOT: 8.40 ms
Average latency: 1.12 s
P50 latency: 1.09 s
P95 latency: 1.74 s
P99 latency: 1.95 s
------------------------
--- Benchmark Results ---
Total time: 26.03s
Requests sent: 500
Throughput: 22131.58 tokens/s
Average TTFT: 128.00 ms
P50 TTFT: 100.39 ms
P95 TTFT: 322.80 ms
P99 TTFT: 386.04 ms
Average TPOT: 16.51 ms
Average latency: 2.22 s
P50 latency: 2.02 s
P95 latency: 3.70 s
P99 latency: 3.83 s
"""
