import torch
import transformers

from tinyquant.utils import quantize_matching_linear_layers


def main() -> None:
    model_id = "unsloth/Llama-3.2-1B"
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    device = model.get_input_embeddings().weight.device

    # One-line quantization
    quantize_matching_linear_layers(model, "nf4", "model.layers.*.self_attn.q_proj")

    prompt = "Quantization for neural networks helps with "
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    output = model.generate(inputs, do_sample=True, max_new_tokens=100)
    print("Quantized model generation:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
