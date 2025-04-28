# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# device = "mps" if torch.backends.mps.is_available() else "cpu"

# # Load fine-tuned model
# model_path = "./fine_tuned_model"
# model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# def predict(question, options):
#     input_text = f"Question: {question}\nOptions: {', '.join(options)}"
#     inputs = tokenizer(input_text, return_tensors="pt").to(device)  # Move inputs to MPS
    
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_length=512)  # Use `generate()` for causal LM
        
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return generated_text  # Return the generated answer

# # Example usage
# if __name__ == "__main__":
#     sample_question = "What is the primary function of the liver?"
#     sample_options = ["A. Produces insulin", "B. Detoxifies blood", "C. Pumps blood", "D. Aids in digestion"]
#     print("Prediction:", predict(sample_question, sample_options))


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load fine-tuned model
model_path = "./fine_tuned_model"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def predict(question, options):
    input_text = f"Question: {question}\nOptions:\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + 5)  # Generate a short answer

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the last token from the generated text
    answer_token = generated_text.split("Answer:")[-1].strip()

    # Map numerical or letter predictions back to options
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "0": 0, "1": 1, "2": 2, "3": 3}
    
    if answer_token in answer_map:
        return options[answer_map[answer_token]]
    else:
        return f"Unrecognized answer: {answer_token}"  # Fallback for debugging

# Example usage
if __name__ == "__main__":
    sample_question_1 = "What is the primary function of red blood cells?"
    sample_options_1 = ["Transport oxygen", "Produce antibodies", "Regulate body temperature", "Digest food"]
    print("Prediction:", predict(sample_question_1, sample_options_1))

    sample_question_2 = "Which organ is responsible for filtering blood in the human body?"
    sample_options_2 = ["Liver", "Heart", "Kidney", "Lungs"]
    print("Prediction:", predict(sample_question_2, sample_options_2))

    sample_question_3 = "Which vitamin is essential for blood clotting?"
    sample_options_3 = ["Vitamin A", "Vitamin C", "Vitamin K", "Vitamin D"]
    print("Prediction:", predict(sample_question_3, sample_options_3))

    sample_question_4 = "What is the primary function of the pancreas?"
    sample_options_4 = ["Produces insulin", "Stores bile", "Regulates heartbeat", "Produces red blood cells"]
    print("Prediction:", predict(sample_question_4, sample_options_4))

    sample_question_5 = "Which type of blood vessel carries oxygen-rich blood away from the heart?"
    sample_options_5 = ["Veins", "Arteries", "Capillaries", "Lymphatics"]
    print("Prediction:", predict(sample_question_5, sample_options_5))

    sample_question_6 = "What is the largest organ in the human body?"
    sample_options_6 = ["Heart", "Liver", "Skin", "Brain"]
    print("Prediction:", predict(sample_question_6, sample_options_6))

    sample_question_7 = "Which part of the brain controls balance and coordination?"
    sample_options_7 = ["Cerebrum", "Cerebellum", "Medulla", "Hypothalamus"]
    print("Prediction:", predict(sample_question_7, sample_options_7))

    sample_question_8 = "Which of the following is a bacterial infection?"
    sample_options_8 = ["Malaria", "Tuberculosis", "Influenza", "Chickenpox"]
    print("Prediction:", predict(sample_question_8, sample_options_8))

    sample_question_9 = "Which hormone regulates blood sugar levels?"
    sample_options_9 = ["Adrenaline", "Insulin", "Thyroxine", "Glucagon"]
    print("Prediction:", predict(sample_question_9, sample_options_9))

    sample_question_10 = "Which condition is caused by a deficiency of iron?"
    sample_options_10 = ["Osteoporosis", "Anemia", "Diabetes", "Hypertension"]
    print("Prediction:", predict(sample_question_10, sample_options_10))

    sample_question_11 = "What is the normal pH range of human blood?"
    sample_options_11 = ["6.0 - 6.5", "7.35 - 7.45", "8.0 - 8.5", "5.5 - 6.0"]
    print("Prediction:", predict(sample_question_11, sample_options_11))

    sample_question_12 = "Which organ produces bile?"
    sample_options_12 = ["Pancreas", "Liver", "Stomach", "Gallbladder"]
    print("Prediction:", predict(sample_question_12, sample_options_12))

    sample_question_13 = "Which type of white blood cell is responsible for producing antibodies?"
    sample_options_13 = ["Monocytes", "Lymphocytes", "Neutrophils", "Eosinophils"]
    print("Prediction:", predict(sample_question_13, sample_options_13))

    sample_question_14 = "What is the most common type of cancer worldwide?"
    sample_options_14 = ["Lung cancer", "Breast cancer", "Skin cancer", "Colon cancer"]
    print("Prediction:", predict(sample_question_14, sample_options_14))

    sample_question_15 = "Which part of the eye controls the amount of light entering it?"
    sample_options_15 = ["Cornea", "Lens", "Pupil", "Retina"]
    print("Prediction:", predict(sample_question_15, sample_options_15))

