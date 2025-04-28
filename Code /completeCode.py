import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import Trainer, DataCollatorWithPadding

model_name = "../Llama-3.2-1B"  # Replace with your model's path

# Load model 
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None  # Disable auto device mapping
).to(device)

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



# Load tokenizer (adjust model name if needed)
tokenizer = AutoTokenizer.from_pretrained("../Llama-3.2-1B")
# Set padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  




# Load the dataset

dataset = load_dataset("medmcqa", split={"train": "train", "test": "test", "validation": "validation"})

subset_size = 8000   # Reduce dataset size for optimized training
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(subset_size))

if tokenizer.pad_token is None:  # Tokenization
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
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()

    return tokenized_inputs

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)
# (Optional) Save as a Hugging Face dataset format
tokenized_dataset.save_to_disk("medmcqa_tokenized")
print("Tokenized dataset saved in Hugging Face dataset format!")


# Training

training_args = TrainingArguments(
    output_dir="./results",         
    per_device_train_batch_size=1,    # Keep batch size low due to 8GB RAM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,   # Accumulate gradients to simulate larger batches
    learning_rate=1e-4,               # Adjust based on performance
    num_train_epochs=1,               # Number of epochs (adjustable)
    logging_dir="./logs",             # Where to store logs
    logging_steps=50,                 # Log training info every 50 steps
    save_strategy="epoch",            # Save model at the end of each epoch
    eval_strategy="no",               # Evaluate at the end of each epoch
    fp16=False,                     
)

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

print("Fine-tuning completed and model saved!")

