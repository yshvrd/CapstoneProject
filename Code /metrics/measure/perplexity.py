# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import math

# def calculate_perplexity(model, tokenizer, text):
#     inputs = tokenizer(text, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs, labels=inputs["input_ids"])
#     loss = outputs.loss.item()
#     perplexity = math.exp(loss)
#     return perplexity

# if __name__ == "__main__":
#     model_path = "../Llama-3.2-1B"  # Change to base model path to compare
#     model = AutoModelForCausalLM.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     test_text = "The quick brown fox jumps over the lazy dog."
#     ppl = calculate_perplexity(model, tokenizer, test_text)
#     print(f"Perplexity: {ppl:.2f}")



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import matplotlib.pyplot as plt

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return math.exp(loss)

if __name__ == "__main__":
    model_path = "../Llama-3.2-1B"  # Change this to your base model path
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # List of 5 different questions for evaluation
    test_texts = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "How does a neural network learn?",
        "What are the benefits of exercise?",
        "Describe the process of photosynthesis."
    ]

    perplexities = [calculate_perplexity(model, tokenizer, text) for text in test_texts]
    avg_perplexity = sum(perplexities) / len(perplexities)

    # Plot the line graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(test_texts)), perplexities, marker='o', linestyle='-', color='b', label="Base Model Perplexity")
    
    # Formatting the graph
    plt.xticks(range(len(test_texts)), [f"Q{i+1}" for i in range(len(test_texts))], rotation=30)
    plt.xlabel("Questions")
    plt.ylabel("Perplexity")
    plt.title(f"Model Perplexity on Different Questions\n(Average: {avg_perplexity:.2f})")
    plt.ylim(0, max(perplexities) + 5)
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Perplexities: {perplexities}")
    print(f"Average Perplexity: {avg_perplexity:.2f}")
