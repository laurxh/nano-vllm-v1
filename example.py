import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.abspath(os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"模型目录不存在: {path}\n"
            "请确认路径是否正确，或把 example.py 里的 path 改成你的本地模型目录。"
        )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    llm = LLM(path, enforce_eager=True, max_model_len=4096)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
