from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(model, tokenizer, prompt, reference):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    bleu_score = sentence_bleu([reference.split()], generated_text.split())
    return bleu_score, generated_text

if __name__ == "__main__":
    model_path = "../Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = "What is artificial intelligence?"
    reference_text = "Artificial intelligence is the simulation of human intelligence in machines."

    bleu, output = calculate_bleu(model, tokenizer, prompt, reference_text)
    print(f"Generated: {output}")
    print(f"BLEU Score: {bleu:.2f}")
