import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="cpu"
    )

prompt = "What is the normal range of human body temperature?"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_length=50)

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\nQuestion: ", prompt, "\n")
print("Model Response:", response, "\n")
