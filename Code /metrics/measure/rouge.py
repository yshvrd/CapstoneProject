from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer

def calculate_rouge(model, tokenizer, prompt, reference):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated_text)
    
    return scores, generated_text

if __name__ == "__main__":
    model_path = "../Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = "Explain quantum computing."
    reference_text = "Quantum computing is a type of computing that uses qubits instead of bits."

    rouge_scores, output = calculate_rouge(model, tokenizer, prompt, reference_text)
    print(f"Generated: {output}")
    print(f"ROUGE Scores: {rouge_scores}")
