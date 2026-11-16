from transformers import AutoTokenizer, AutoModelForCausalLM

name = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

print("Parameters:", model.num_parameters())

inputs = tokenizer("Once upon a time", return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.8
)