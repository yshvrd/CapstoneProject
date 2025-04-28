from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Load tokenizer (adjust model name if needed)
tokenizer = AutoTokenizer.from_pretrained("../Llama-3.2-1B")

# Set padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  
# Load the dataset
dataset = load_dataset("medmcqa", split={"train": "train", "test": "test", "validation": "validation"})
print("MedMCQA dataset loaded successfully!")

# Reduce dataset size for optimized training
subset_size = 10000  # Change this if needed
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(subset_size))
print(f"Using a subset of {subset_size} training examples.")

# Tokenization
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token

def preprocess_function(examples):
    """
    Convert MedMCQA examples into a format suitable for causal language modeling,
    ensuring the model learns both the correct answer and an explanation.
    """
    questions = examples["question"]
    options = zip(examples["opa"], examples["opb"], examples["opc"], examples["opd"])
    answers = examples["cop"]  # Correct option (e.g., 'A', 'B', 'C', 'D')
    explanations = examples["exp"]  # Explanation field

    # Format data to include both the correct answer and an explanation
    inputs = [
        f"Question: {q}\nOptions:\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer: {ans}\nExplanation: {exp}"  
        for q, (a, b, c, d), ans, exp in zip(questions, options, answers, explanations)
    ]

    # Tokenize the inputs
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

    # Set labels as input_ids (shifted right)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    
    return tokenized_inputs


# Apply tokenization to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)
print("Dataset tokenized successfully!")

# (Optional) Save as a Hugging Face dataset format
tokenized_dataset.save_to_disk("medmcqa_tokenized")
print("Tokenized dataset saved in Hugging Face dataset format!")
