# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# # Load base model
# base_model_name = "../Llama-3.2-1B"
# model = AutoModelForCausalLM.from_pretrained(base_model_name)

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# # Load fine-tuned LoRA adapter
# fine_tuned_model_path = "./results"
# model = PeftModel.from_pretrained(model, fine_tuned_model_path)

# # Move to device (if available)
# import torch
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# model.to(device)

# def generate_answer(question):
#     inputs = tokenizer(question, return_tensors="pt").to(device)
#     output = model.generate(**inputs, max_length=100)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # Test with a sample MedMCQA question
# question = "What is the normal range of human body temperature?"
# print(generate_answer(question))


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model_name = "../Llama-3.2-1B"  
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token  # Prevents errors during padding
tokenizer.padding_side = "left"

# Load fine-tuned LoRA adapter
fine_tuned_model_path = "./results/checkpoint-500"  # Ensure this path is correct
model = PeftModel.from_pretrained(model, fine_tuned_model_path)

# Apply LoRA weights
# model = model.merge_and_unload()  # Ensures fine-tuned weights are applied

# Move model to correct device (MPS for Mac, else CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Function to generate answers
def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            min_new_tokens = 50,
            max_length=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,  # Prevents errors
            do_sample=True,  # Enable randomness in generation
            temperature=0.4,  # Controls randomness (lower = more deterministic)
            top_p=0.8,  # Nucleus sampling for diverse responses
            repetition_penalty=1.6,
            penalty_alpha=0.6,  # Discourages repeating input
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test with a sample MedMCQA question
question1 = "What is the normal range of human body temperature?"
print("\nQuestion:", question1)
print("Answer:", generate_answer(question1), "\n")

question2 = "Which neurotransmitter is primarily involved in Parkinson's disease?"
print("\nQuestion:", question2)
print("Answer:", generate_answer(question2), "\n")
