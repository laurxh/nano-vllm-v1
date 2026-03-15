# nano-vllm-v1 Benchmark 实验记录

## 测试命令

```bash
CUDA_VISIBLE_DEVICES=7 python serving_bench.py \
  --model ~/huggingface/Qwen3-0.6B \
  --num-requests 500 \
  --request-rate 20 \
  --random-input-len 1024 \
  --random-output-len 128 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 256 \
  --chunked-prefill
```

使用以上参数运行 `nano-vllm-v1`，得到如下实验结果。

## Benchmark 构造方式

本实验使用 `serving_bench.py` 构造在线服务场景下的请求流，并对不同调度策略进行对比。

### 请求构造

- 模型使用 `~/huggingface/Qwen3-0.6B`
- 请求总数为 `500`
- 每个请求的输入长度固定为 `1024` token
- 每个请求的输出长度固定为 `128` token
- 输入 token 由随机整数生成，用于模拟长度一致的合成请求
- 采样参数使用 `ignore_eos=True`，因此每个请求会尽量生成到设定的最大输出长度

### 到达过程

- 请求到达服从指数分布
- 平均到达速率为 `20 requests/s`
- 脚本通过累积到达时间 `arrival_times`，逐步将请求送入引擎
- 这种构造方式近似模拟在线推理服务中的泊松到达流量

### 调度与运行参数

- `max-num-batched-tokens=2048`
- `max-num-seqs=256`
- 开启 `chunked-prefill` 时，请求的 prefill 阶段会按当前 step 可用的 token budget 分段执行
- 未开启 `chunked-prefill` 时，请求会按照对应实现的默认 prefill / decode 调度逻辑执行

### 统计指标

- `Throughput`：总处理 token 数除以总耗时
- `TTFT`：请求提交到首 token 返回的时间
- `TPOT`：首 token 之后，平均每个输出 token 的生成时间
- `Latency`：请求提交到完整请求结束的总时延
- 文中同时统计了平均值、P50、P95 和 P99

## 实验结果

### 原版

```text
--- Benchmark Results ---
Total time: 26.69s
Requests sent: 500
Throughput: 21580.56 tokens/s
Average TTFT: 39.44 ms
P50 TTFT: 36.00 ms
P95 TTFT: 72.48 ms
P99 TTFT: 108.34 ms
Average TPOT: 28.16 ms
Average latency: 3.62 s
P50 latency: 3.27 s
P95 latency: 5.42 s
P99 latency: 5.50 s
```

### Chunked Prefill

```text
--- Benchmark Results ---
Total time: 25.91s
Requests sent: 500
Throughput: 22233.18 tokens/s
Average TTFT: 57.34 ms
P50 TTFT: 52.77 ms
P95 TTFT: 94.36 ms
P99 TTFT: 111.16 ms
Average TPOT: 8.56 ms
Average latency: 1.14 s
P50 latency: 1.11 s
P95 latency: 1.74 s
P99 latency: 2.01 s
```

### 普通 PD 混合，非 Chunked Prefill

```text
--- Benchmark Results ---
Total time: 25.95s
Requests sent: 500
Throughput: 22193.24 tokens/s
Average TTFT: 94.44 ms
P50 TTFT: 76.06 ms
P95 TTFT: 207.62 ms
P99 TTFT: 270.78 ms
Average TPOT: 12.03 ms
Average latency: 1.62 s
P50 latency: 1.53 s
P95 latency: 2.74 s
P99 latency: 3.07 s
```

## 简要观察

- 在这组负载下，`chunked-prefill` 的吞吐最高，为 `22233.18 tokens/s`
- 原版的 `TTFT` 最低，但 `TPOT` 和整体 `latency` 明显更高
- `chunked-prefill` 在 `latency`、`TPOT` 和尾部时延上表现最好
- 普通 PD 混合方案的整体表现介于原版和 `chunked-prefill` 之间
