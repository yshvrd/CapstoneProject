import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments




model_name = "Llama-3.2-1B"  # Replace with your model's path

# Load model in mixed precision (bfloat16) for macOS
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model loaded successfully!")


# Configure LoRA (Low-Rank Adaptation)
lora_config = LoraConfig(
    r=8,  # Low-rank size (increase for better adaptation, but requires more RAM)
    lora_alpha=32,  
    lora_dropout=0.05,  
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
print("LoRA enabled successfully!")


# # Load MedMCQA dataset from Hugging Face
# dataset = load_dataset("medmcqa")

# print(dataset)
# print("MedMCQA dataset loaded successfully!")

# Load the dataset
dataset = load_dataset("medmcqa", split={"train": "train", "test": "test", "validation": "validation"})
print("MedMCQA dataset loaded successfully!")

# Reduce dataset size for optimized training
subset_size = 10000  # Change this to 50000 if needed
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(subset_size))

print(f"Using a subset of {subset_size} training examples.")

# Proceed with tokenization...



# Tokenization 
# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token
def preprocess_function(examples):
    """
    Convert MedMCQA examples into a format suitable for causal language modeling.
    """
    questions = examples["question"]
    options = zip(examples["opa"], examples["opb"], examples["opc"], examples["opd"])
    answers = examples["cop"]  # Correct option (e.g., 'A', 'B', 'C', 'D')

    # Format data as "Question: ... Options: A) ... B) ... C) ... D) ... Answer: ..."
    inputs = [
        f"Question: {q}\nOptions:\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:"
        for q, (a, b, c, d) in zip(questions, options)
    ]

    # Tokenize the inputs
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    
    return tokenized_inputs

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

print("Dataset tokenized successfully!")



# Trainign 

training_args = TrainingArguments(
    output_dir="./results",         # Where to save the model
    per_device_train_batch_size=1,  # Keep batch size low due to 8GB RAM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batches
    learning_rate=2e-4,             # Adjust based on performance
    num_train_epochs=3,             # Number of epochs (adjustable)
    logging_dir="./logs",           # Where to store logs
    logging_steps=50,               # Log training info every 50 steps
    save_strategy="epoch",          # Save model at the end of each epoch
    evaluation_strategy="epoch",    # Evaluate at the end of each epoch
    fp16=False,                     # Mac M1 doesn't support native FP16
    push_to_hub=False               # Disable pushing to Hugging Face Hub
)
from transformers import Trainer, DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)


trainer.train()
trainer.evaluate()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

def predict(question, options):
    input_text = f"Question: {question} Options: {options}"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return options[predicted_class]

# Example usage
sample_question = "What is the primary function of the liver?"
sample_options = ["A. Produces insulin", "B. Detoxifies blood", "C. Pumps blood", "D. Aids in digestion"]

print("Prediction:", predict(sample_question, sample_options))
