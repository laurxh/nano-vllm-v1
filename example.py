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

"""
Prompt: '<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n'
Completion: "<think>\nOkay, the user wants me to introduce myself. Let me start by recalling my training and experiences. I've been training for years, working in different areas like customer service, project management, and digital marketing. I've worked in various industries, so I can mention a few examples. I should be friendly and open to learning more. I need to keep it simple and concise, avoiding any technical jargon. Also, make sure to highlight my skills and how I can help the user. Let me check if I have any specific details I should include, like my name, experience, or areas of expertise. I think that's all. Now, I'll put it all together in a natural way.\n</think>\n\nHello! I'm [Your Name], and I've been training in various fields like customer service, project management, and digital marketing for years. I've worked in different industries, and I'm passionate about helping others in these areas. I'm always looking to learn and grow, so feel free to ask questions! 😊<|im_end|>"
"""
