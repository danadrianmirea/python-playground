
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tiny GPT-2 model and tokenizer
model_name = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model loaded successfully!")

# Ask a question
#prompt = "Question: What is Python?\nAnswer:"
prompt = "Hello"

# Convert the prompt into tokens
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id
)

# Convert generated tokens back into readable text
response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nModel response:")
print(response)
