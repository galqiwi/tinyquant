import torch
import transformers

from tinyquant.utils import quantize_matching_linear_layers
from tinyquant.integrations.quarot.quarot2 import apply_quarot_rotation

from tinyquant.integrations.quarot.quarot import (
    QuaRotDataFreeConfig,
    apply_quarot_datafree_llama,
)



def main() -> None:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    # quantize_matching_linear_layers(model, "nf4", "model.layers.*.self_attn.q_proj")

    # apply_quarot_rotation(model, mode="hadamard")

    # quantize_matching_linear_layers(
    #     model,
    #     method_name="nf4",
    #     pattern="model.layers.*.self_attn.q_proj",
    # )

    cfg = QuaRotDataFreeConfig(
        w_bits=4,
        w_asym=False,
        w_groupsize=-1,
        int8_down_proj=False,
        rotate=True,
        rotate_mode="hadamard",
        fp32_had=False,
    )

    apply_quarot_datafree_llama(model, cfg)

    model.to('cuda')
    output = model.generate(
        tokenizer("My name is ", return_tensors="pt")["input_ids"].cuda(), max_new_tokens=1000, do_sample=True
    )
    print(tokenizer.decode(output[0]))


if __name__ == "__main__":
    main()
