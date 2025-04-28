# import torch
# import matplotlib.pyplot as plt
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from rouge_score import rouge_scorer

# # Define model paths
# base_model_path = "./Llama-3.2-1B"
# fine_tuned_model_path = "./results/checkpoint-250"

# # Load models & tokenizers
# base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
# base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
# fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# # Define test questions and reference answers
# test_texts = [
#     "What is the primary function of the liver in metabolism?",
#     "How does insulin regulate blood sugar levels?",
#     "What are the common symptoms of meningitis?",
#     "What class of drugs is commonly used to treat hypertension?",
#     "What is the difference between incidence and prevalence in disease studies?"
# ]

# reference_answers = [
#     "The liver processes nutrients, detoxifies harmful substances, and produces bile.",
#     "Insulin helps glucose enter cells, reducing blood sugar levels.",
#     "Common symptoms include fever, headache, stiff neck, and confusion.",
#     "Antihypertensive drugs include beta-blockers, ACE inhibitors, and diuretics.",
#     "Incidence measures new cases; prevalence measures total cases at a given time."
# ]

# # Define text generation function
# def generate_text(model, tokenizer, prompt):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_length=50)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Initialize ROUGE scorer
# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# # Compute ROUGE scores
# base_rouge1_scores = []
# fine_tuned_rouge1_scores = []

# for i, question in enumerate(test_texts):
#     base_answer = generate_text(base_model, base_tokenizer, question)
#     fine_tuned_answer = generate_text(fine_tuned_model, fine_tuned_tokenizer, question)

#     base_rouge = scorer.score(reference_answers[i], base_answer)
#     fine_tuned_rouge = scorer.score(reference_answers[i], fine_tuned_answer)

#     base_rouge1_scores.append(base_rouge['rouge1'].fmeasure)
#     fine_tuned_rouge1_scores.append(fine_tuned_rouge['rouge1'].fmeasure)

# # Plot ROUGE scores
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, len(test_texts) + 1), base_rouge1_scores, marker='o', linestyle='-', label="Base Model ROUGE-1")
# plt.plot(range(1, len(test_texts) + 1), fine_tuned_rouge1_scores, marker='s', linestyle='-', label="Fine-Tuned Model ROUGE-1")
# plt.xlabel("Question Number")
# plt.ylabel("ROUGE-1 Score")
# plt.title("ROUGE-1 Score Comparison: Base vs. Fine-Tuned Model")
# plt.legend()
# plt.grid()
# plt.show()




import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to generate text
def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to calculate ROUGE scores
def calculate_rouge(model, tokenizer, questions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    rouge1_scores, rouge2_scores = [], []

    for question, reference in zip(questions, references):
        generated_text = generate_text(model, tokenizer, question)
        scores = scorer.score(reference, generated_text)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)

    return rouge1_scores, rouge2_scores

# Load Base and Fine-Tuned Models
base_model_path = "./Llama-3.2-1B"  # Change if needed
fine_tuned_model_path = "./results/checkpoint-250"

base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# Sample medical questions and reference answers
questions = [
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

references = [
    "Myocardial infarction is primarily caused by a blockage in the coronary arteries due to atherosclerosis or a thrombus.",
    "Insulin lowers blood glucose by promoting cellular uptake of glucose and inhibiting gluconeogenesis in the liver.",
    "Symptoms of meningitis include fever, headache, neck stiffness, photophobia, and altered mental status.",
    "Hypertension results from increased vascular resistance due to arterial stiffness, endothelial dysfunction, or excessive sympathetic activity.",
    "Long-term corticosteroid use can cause osteoporosis, hyperglycemia, immunosuppression, muscle wasting, and adrenal suppression.",
    "Beta-blockers work by inhibiting beta-adrenergic receptors, reducing heart rate, myocardial oxygen demand, and blood pressure.",
    "Proton pump inhibitors block gastric H+/K+ ATPase, reducing acid secretion and promoting esophageal mucosal healing.",
    "N-acetylcysteine (NAC) replenishes glutathione stores, enhancing acetaminophen detoxification in the liver.",
    "First-generation antihistamines cross the blood-brain barrier causing sedation, while second-generation ones are non-sedating and selective.",
    "NSAIDs inhibit prostaglandin synthesis, reducing gastric mucosal protection and increasing ulcer risk.",
    "MRI with contrast is the preferred imaging modality for detecting brain tumors due to high resolution and soft tissue differentiation.",
    "STEMI presents with ST elevation, while NSTEMI shows ST depression or T-wave inversions.",
    "Diabetes is diagnosed using fasting blood glucose, HbA1c levels, and oral glucose tolerance tests.",
    "Chest X-ray can show lung consolidation, alveolar infiltrates, and air bronchograms, indicative of pneumonia.",
    "MRI findings include periventricular white matter lesions, Dawson’s fingers, and contrast-enhancing plaques.",
    "Hypertension, diabetes, atrial fibrillation, smoking, and hyperlipidemia are major risk factors for stroke.",
    "Cirrhosis leads to fibrosis, portal hypertension, reduced hepatic metabolism, and impaired coagulation.",
    "Bradykinesia, resting tremor, rigidity, and postural instability are key features of Parkinson’s disease.",
    "Autoimmune hemolytic anemia is due to antibody-mediated RBC destruction, while sickle cell disease is caused by abnormal hemoglobin polymerization.",
    "Tumor markers like PSA, CA-125, and AFP aid in diagnosis, prognosis, and monitoring cancer recurrence."
]


# Calculate ROUGE scores
base_rouge1, base_rouge2 = calculate_rouge(base_model, base_tokenizer, questions, references)
fine_tuned_rouge1, fine_tuned_rouge2 = calculate_rouge(fine_tuned_model, fine_tuned_tokenizer, questions, references)

# Plot ROUGE-1
plt.figure(figsize=(10, 5))
plt.plot(questions, base_rouge1, marker='o', linestyle='-', label="Base Model")
plt.plot(questions, fine_tuned_rouge1, marker='s', linestyle='--', label="Fine-Tuned Model")
plt.xticks(rotation=25)
plt.xlabel("Question")
plt.ylabel("ROUGE-1 Score")
plt.suptitle("ROUGE-1 Comparison(Unigram Overlap)", fontsize=14)
plt.title("Higher is Better", fontsize=10, color='#228B22')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)  # ROUGE scores are between 0 and 1
plt.show()

# Plot ROUGE-2
plt.figure(figsize=(10, 5))
plt.plot(questions, base_rouge2, marker='o', linestyle='-', label="Base Model")
plt.plot(questions, fine_tuned_rouge2, marker='s', linestyle='--', label="Fine-Tuned Model")
plt.xticks(rotation=25)
plt.xlabel("Question")
plt.ylabel("ROUGE-2 Score")
plt.suptitle("ROUGE-2 Comparison(Bigram Overlap)", fontsize=14)
plt.title("Higher is Better", fontsize=10, color='#228B22')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)  # ROUGE scores are between 0 and 1
plt.show()
