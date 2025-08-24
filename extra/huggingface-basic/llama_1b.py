import torch
import transformers

from tinyquant.utils import quantize_matching_linear_layers


def main() -> None:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    quantize_matching_linear_layers(model, "nf4", "model.layers.*.self_attn.q_proj")
    output = model.generate(
        tokenizer("", return_tensors="pt")["input_ids"].cuda(), max_new_tokens=1000
    )
    print(tokenizer.decode(output[0]))


if __name__ == "__main__":
    main()
