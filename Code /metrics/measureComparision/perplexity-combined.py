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
    # Define model paths
    base_model_path = "./Llama-3.2-1B"  # Update if needed
    fine_tuned_model_path = "./results/checkpoint-250"

    # Load models and tokenizers
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

    # List of test questions
    test_texts = [
    "What is the primary cause of myocardial infarction?",
    "How does insulin regulate blood glucose levels?",
    "What are the common symptoms of meningitis?",
    "Describe the pathophysiology of hypertension.",
    "What are the side effects of prolonged corticosteroid use?",
    "What is the mechanism of action of beta-blockers?",
    "How do proton pump inhibitors help in treating GERD?",
    "What is the antidote for acetaminophen toxicity?",
    "Compare first-generation and second-generation antihistamines.",
    "Why should NSAIDs be avoided in patients with gastric ulcers?",
    "What imaging technique is best for detecting a brain tumor?",
    "How do ECG findings differ in STEMI vs. NSTEMI?",
    "What lab tests confirm a diagnosis of diabetes mellitus?",
    "How can a chest X-ray help diagnose pneumonia?",
    "What are the characteristic MRI findings in multiple sclerosis?",
    "What are the major risk factors for stroke?",
    "How does cirrhosis affect liver function?",
    "What are the hallmark signs of Parkinson’s disease?",
    "How does autoimmune hemolytic anemia differ from sickle cell disease?",
    "What is the role of tumor markers in cancer diagnosis?"
]


    # Compute perplexities for both models
    base_perplexities = [calculate_perplexity(base_model, base_tokenizer, text) for text in test_texts]
    fine_tuned_perplexities = [calculate_perplexity(fine_tuned_model, fine_tuned_tokenizer, text) for text in test_texts]

    avg_base_ppl = sum(base_perplexities) / len(base_perplexities)
    avg_fine_tuned_ppl = sum(fine_tuned_perplexities) / len(fine_tuned_perplexities)

    # Plot the line graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(test_texts)), base_perplexities, marker='o', linestyle='-', color='b', label="Base Model")
    plt.plot(range(len(test_texts)), fine_tuned_perplexities, marker='s', linestyle='--', color='r', label="Fine-Tuned Model")

    # Formatting the graph
    plt.xticks(range(len(test_texts)), [f"Q{i+1}" for i in range(len(test_texts))], rotation=30)
    plt.xlabel("Questions")
    plt.ylabel("Perplexity Score")
    plt.suptitle("Perplexity", fontsize=14)
    plt.title("Lower is Better", fontsize=10, color='#228B22')
    plt.ylim(0, max(base_perplexities + fine_tuned_perplexities) + 5)
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Base Model Perplexities: {base_perplexities}")
    print(f"Fine-Tuned Model Perplexities: {fine_tuned_perplexities}")
    print(f"Average Base Model Perplexity: {avg_base_ppl:.2f}")
    print(f"Average Fine-Tuned Model Perplexity: {avg_fine_tuned_ppl:.2f}")
