import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Define model paths
base_model_path = "./Llama-3.2-1B"
fine_tuned_model_path = "./results/checkpoint-250"

# Load models & tokenizers
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# Define test questions and reference answers
# test_texts = [
#     "What is the primary cause of myocardial infarction?",
#     "How does insulin regulate blood glucose levels?",
#     "What are the common symptoms of meningitis?",
#     "Describe the pathophysiology of hypertension.",
#     "What are the side effects of prolonged corticosteroid use?",
#     "What is the mechanism of action of beta-blockers?",
#     "How do proton pump inhibitors help in treating GERD?",
#     "What is the antidote for acetaminophen toxicity?",
#     "Compare first-generation and second-generation antihistamines.",
#     "Why should NSAIDs be avoided in patients with gastric ulcers?",
#     "What imaging technique is best for detecting a brain tumor?",
#     "How do ECG findings differ in STEMI vs. NSTEMI?",
#     "What lab tests confirm a diagnosis of diabetes mellitus?",
#     "How can a chest X-ray help diagnose pneumonia?",
#     "What are the characteristic MRI findings in multiple sclerosis?",
#     "What are the major risk factors for stroke?",
#     "How does cirrhosis affect liver function?",
#     "What are the hallmark signs of Parkinson’s disease?",
#     "How does autoimmune hemolytic anemia differ from sickle cell disease?",
#     "What is the role of tumor markers in cancer diagnosis?"
# ]

# reference_answers = [
#     "Myocardial infarction is primarily caused by a blockage in the coronary arteries due to atherosclerosis or a thrombus.",
#     "Insulin lowers blood glucose by promoting cellular uptake of glucose and inhibiting gluconeogenesis in the liver.",
#     "Symptoms of meningitis include fever, headache, neck stiffness, photophobia, and altered mental status.",
#     "Hypertension results from increased vascular resistance due to arterial stiffness, endothelial dysfunction, or excessive sympathetic activity.",
#     "Long-term corticosteroid use can cause osteoporosis, hyperglycemia, immunosuppression, muscle wasting, and adrenal suppression.",
#     "Beta-blockers work by inhibiting beta-adrenergic receptors, reducing heart rate, myocardial oxygen demand, and blood pressure.",
#     "Proton pump inhibitors block gastric H+/K+ ATPase, reducing acid secretion and promoting esophageal mucosal healing.",
#     "N-acetylcysteine (NAC) replenishes glutathione stores, enhancing acetaminophen detoxification in the liver.",
#     "First-generation antihistamines cross the blood-brain barrier causing sedation, while second-generation ones are non-sedating and selective.",
#     "NSAIDs inhibit prostaglandin synthesis, reducing gastric mucosal protection and increasing ulcer risk.",
#     "MRI with contrast is the preferred imaging modality for detecting brain tumors due to high resolution and soft tissue differentiation.",
#     "STEMI presents with ST elevation, while NSTEMI shows ST depression or T-wave inversions.",
#     "Diabetes is diagnosed using fasting blood glucose, HbA1c levels, and oral glucose tolerance tests.",
#     "Chest X-ray can show lung consolidation, alveolar infiltrates, and air bronchograms, indicative of pneumonia.",
#     "MRI findings include periventricular white matter lesions, Dawson’s fingers, and contrast-enhancing plaques.",
#     "Hypertension, diabetes, atrial fibrillation, smoking, and hyperlipidemia are major risk factors for stroke.",
#     "Cirrhosis leads to fibrosis, portal hypertension, reduced hepatic metabolism, and impaired coagulation.",
#     "Bradykinesia, resting tremor, rigidity, and postural instability are key features of Parkinson’s disease.",
#     "Autoimmune hemolytic anemia is due to antibody-mediated RBC destruction, while sickle cell disease is caused by abnormal hemoglobin polymerization.",
#     "Tumor markers like PSA, CA-125, and AFP aid in diagnosis, prognosis, and monitoring cancer recurrence."
# ]



test_texts = [
    "What is the normal human body temperature?",
    "How many chambers does the human heart have?",
    "What organ produces insulin?",
    "What is the main function of red blood cells?",
    "What is the medical term for high blood sugar?",
    "What is the largest organ in the human body?",
    "What vitamin is produced when the skin is exposed to sunlight?",
    "What is the most common symptom of the flu?",
    "Which bone is commonly known as the collarbone?",
    "What part of the body does an ophthalmologist specialize in?",
    "What is the pathophysiology of Guillain-Barré syndrome?",
    "Which scoring system is used to assess mortality risk in ICU patients?",
    "What is the mechanism of action of pembrolizumab?",
    "What are the Koch’s postulates in microbiology?",
    "Which autoantibody is highly specific for systemic lupus erythematosus (SLE)?",
    "What is the primary site of action of thiazide diuretics in the kidney?",
    "What is the genetic mutation responsible for Marfan syndrome?",
    "What is the most sensitive cardiac biomarker for detecting myocardial infarction?",
    "What is the first-line antibiotic for treating Lyme disease?",
    "What are the criteria for brain death diagnosis?"
]


reference_answers = [
    "98.6°F (37°C).",
    "Four.",
    "Pancreas.",
    "Oxygen transport.",
    "Hyperglycemia.",
    "Skin.",
    "Vitamin D.",
    "Fever.",
    "Clavicle.",
    "Eyes.",
    "Autoimmune attack on the myelin sheath of peripheral nerves.",
    "APACHE II.",
    "Anti-PD-1 monoclonal antibody blocking immune checkpoints.",
    "Four criteria to establish a causal relationship between a microbe and disease.",
    "Anti-dsDNA.",
    "Distal convoluted tubule.",
    "FBN1 gene mutation.",
    "Troponin.",
    "Doxycycline.",
    "Irreversible coma, absence of brainstem reflexes, and apnea."
]



def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Compute BLEU scores
base_bleu_scores = []
fine_tuned_bleu_scores = []

for i, question in enumerate(test_texts):
    base_answer = generate_text(base_model, base_tokenizer, question)
    fine_tuned_answer = generate_text(fine_tuned_model, fine_tuned_tokenizer, question)

    bleu_base = sentence_bleu(reference_answers[i], base_answer.split(), smoothing_function=SmoothingFunction().method1)
    bleu_fine_tuned = sentence_bleu(reference_answers[i], fine_tuned_answer.split(), smoothing_function=SmoothingFunction().method1)

    base_bleu_scores.append(bleu_base)
    fine_tuned_bleu_scores.append(bleu_fine_tuned)

# Plot BLEU scores
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(test_texts) + 1), base_bleu_scores, marker='o', linestyle='-', label="Base Model BLEU")
plt.plot(range(1, len(test_texts) + 1), fine_tuned_bleu_scores, marker='s', linestyle='-', label="Fine-Tuned Model BLEU")
plt.xlabel("Question")
plt.ylabel("BLEU Score")
plt.suptitle("BLEU Score", fontsize=14)
plt.title("Higher is Better", fontsize=10, color='#228B22')
plt.legend()
plt.grid()
plt.show()
