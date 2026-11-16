
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

print("Model loaded!")
print("Parameters:", model.num_parameters())

while True:
    prompt = input("\nYou: ")

    if prompt.lower() == "exit":
        break

    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated answer
    new_tokens = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    print("\nModel:", response)
