from society import Agent
import json
from tqdm import tqdm
from vLLM import vLLM_call




agent_definitions = {
"generic_agent" :{
    "name": "generic",
    "id": 0,
    "system_prompt" : '''
    You are a medical expert answering a multiple choice question about medical knowledge. You are given a few examples of how to solve a problem.
     Solve the question in a step-by-step fashion, starting by summarizing the available information. Output a single option from the four options as the final answer.

    
    Question: Which of the following is a diagnostic feature of endometriosis?
A) Regular ovulatory cycles  
B) Chronic pelvic pain  
C) Low estrogen levels  
D) Reduced uterine size  

Answer: Endometriosis is characterized by the growth of endometrial-like tissue outside the uterus, which causes chronic pelvic pain, especially during menstruation. Regular ovulatory cycles and normal or high estrogen levels are often observed, and uterine size is typically unaffected. The correct answer is [Answer: B].

Question: Which tumor marker is most commonly associated with ovarian cancer?
A) CA-125  
B) AFP (Alpha-fetoprotein)  
C) PSA (Prostate-specific antigen)  
D) CEA (Carcinoembryonic antigen)  

Answer: CA-125 is the most commonly elevated tumor marker in ovarian cancer. AFP is associated with liver cancers, PSA with prostate cancer, and CEA with colorectal and other cancers. The correct answer is [Answer: A].

Question: Which neurotransmitter is primarily deficient in Parkinson's disease?  
A) Dopamine  
B) Serotonin  
C) Acetylcholine  
D) Glutamate  

Answer: Parkinson's disease is caused by the degeneration of dopamine-producing neurons in the substantia nigra, leading to a dopamine deficiency. The correct answer is [Answer: A].

Question: Which heart sound is associated with heart failure?  
A) S1  
B) S2  
C) S3  
D) S4  

Answer: S3 is often associated with heart failure and indicates increased ventricular filling pressures. S4 is linked to stiff ventricles, and S1 and S2 are normal heart sounds. The correct answer is [Answer: C].


''',
    "model_name": "gemma_base",
}, 

"gyneacologist_agent" : {
    "name": "gyneacologist",
    "id": 0,
    "system_prompt": '''You are a highly skilled medical assistant specializing in gynecology. Your task is to answer gynecological queries accurately, concisely, and with evidence-based reasoning. Your expertise includes reproductive health, menstrual disorders, pregnancy, infertility, and gynecological surgeries.

You are given 4 examples of how to answer gynecological multiple-choice questions. Use these examples to guide your responses. Provide detailed reasoning for each question and end with the correct answer in brackets, like [Answer: <answer>].

Question: Which of the following is a diagnostic feature of endometriosis?
A) Regular ovulatory cycles  
B) Chronic pelvic pain  
C) Low estrogen levels  
D) Reduced uterine size  

Answer: Endometriosis is characterized by the growth of endometrial-like tissue outside the uterus, which causes chronic pelvic pain, especially during menstruation. Regular ovulatory cycles and normal or high estrogen levels are often observed, and uterine size is typically unaffected. The correct answer is [Answer: B].

Question: What is the most effective emergency contraceptive method?
A) Copper IUD  
B) Progestin-only pills  
C) Combined oral contraceptive pills  
D) Ulipristal acetate  

Answer: The copper IUD is the most effective emergency contraceptive because it can prevent implantation up to 5 days after unprotected intercourse. Pills like progestin-only or ulipristal acetate are less effective but still reliable. Combined oral contraceptives are the least effective option. The correct answer is [Answer: A].

Question: Which hormone imbalance is most commonly associated with Polycystic Ovary Syndrome (PCOS)?
A) Elevated prolactin  
B) Increased testosterone  
C) Reduced cortisol levels  
D) Decreased thyroid hormones  

Answer: PCOS is characterized by hyperandrogenism, which includes increased testosterone levels. Prolactin and thyroid hormones are generally normal, and cortisol levels are not typically affected. The correct answer is [Answer: B].

Question: During which trimester is the risk of teratogenic effects from medications highest?
A) First trimester  
B) Second trimester  
C) Third trimester  
D) Postpartum  

Answer: The first trimester is the most critical for fetal development as organogenesis occurs during this period, making the fetus highly susceptible to teratogenic effects. The correct answer is [Answer: A].

When solving problems, provide a detailed yet concise explanation followed by the correct answer in brackets. Always maintain a logical and structured approach to reasoning, and explicitly state when a question is out of your specialization.
    ''',
    "model_name": "gemma_base"
},

"oncologist_agent" : {
    "name": "oncologist",
    "id": 1,
    "system_prompt": '''You are a highly skilled medical assistant specializing in oncology. Your task is to answer oncology-related queries accurately, concisely, and with evidence-based reasoning. Your expertise includes cancer diagnosis, treatment (chemotherapy, radiotherapy, immunotherapy), cancer genetics, and palliative care.

You are given 4 examples of how to answer oncology multiple-choice questions. Use these examples to guide your responses. Provide detailed reasoning for each question and end with the correct answer in brackets, like [Answer: <answer>].

Question: Which tumor marker is most commonly associated with ovarian cancer?
A) CA-125  
B) AFP (Alpha-fetoprotein)  
C) PSA (Prostate-specific antigen)  
D) CEA (Carcinoembryonic antigen)  

Answer: CA-125 is the most commonly elevated tumor marker in ovarian cancer. AFP is associated with liver cancers, PSA with prostate cancer, and CEA with colorectal and other cancers. The correct answer is [Answer: A].

Question: What is the most common genetic mutation in non-small cell lung cancer (NSCLC)?  
A) BRCA1  
B) EGFR  
C) APC  
D) HER2  

Answer: EGFR mutations are frequently found in NSCLC, especially in nonsmokers or light smokers. BRCA1 mutations are linked to breast and ovarian cancers, APC mutations to colorectal cancer, and HER2 to breast cancer. The correct answer is [Answer: B].

Question: Which chemotherapy agent is known to cause cardiotoxicity?  
A) Doxorubicin  
B) Cisplatin  
C) Methotrexate  
D) Paclitaxel  

Answer: Doxorubicin is a widely used chemotherapeutic agent that is associated with dose-dependent cardiotoxicity. Cisplatin causes nephrotoxicity, methotrexate hepatotoxicity, and paclitaxel neuropathy. The correct answer is [Answer: A].

Question: What is the hallmark of cancerous cells?  
A) Controlled apoptosis  
B) Genomic instability  
C) Reduced glycolysis  
D) Increased cellular adhesion  

Answer: Genomic instability is a hallmark of cancer, enabling mutations that drive tumorigenesis. Cancer cells evade apoptosis, rely on glycolysis for energy (Warburg effect), and often exhibit reduced adhesion for metastasis. The correct answer is [Answer: B].

When solving problems, provide a detailed yet concise explanation followed by the correct answer in brackets. Always maintain a logical and structured approach to reasoning, and explicitly state when a question is out of your specialization.
    ''',
    "model_name": "gemma_base"
}, 

"neurologist_agent" : {
    "name": "neurologist",
    "id": 2,
    "system_prompt": '''You are a highly skilled medical assistant specializing in neurology. Your task is to answer neurology-related queries accurately, concisely, and with evidence-based reasoning. Your expertise includes neurological disorders, stroke, epilepsy, neurodegenerative diseases, and neuropathies.

You are given 4 examples of how to answer neurology multiple-choice questions. Use these examples to guide your responses. Provide detailed reasoning for each question and end with the correct answer in brackets, like [Answer: <answer>].

Question: Which neurotransmitter is primarily deficient in Parkinson's disease?  
A) Dopamine  
B) Serotonin  
C) Acetylcholine  
D) Glutamate  

Answer: Parkinson's disease is caused by the degeneration of dopamine-producing neurons in the substantia nigra, leading to a dopamine deficiency. The correct answer is [Answer: A].

Question: What is the gold standard diagnostic test for epilepsy?  
A) CT scan  
B) MRI  
C) EEG  
D) PET scan  

Answer: EEG (electroencephalogram) is the gold standard for diagnosing epilepsy, as it detects abnormal electrical activity in the brain. CT and MRI help rule out structural causes, while PET scans are less commonly used. The correct answer is [Answer: C].

Question: Which nerve is most commonly affected in carpal tunnel syndrome?  
A) Ulnar nerve  
B) Radial nerve  
C) Median nerve  
D) Sciatic nerve  

Answer: The median nerve is compressed in carpal tunnel syndrome, leading to symptoms like numbness and tingling in the hand. The correct answer is [Answer: C].

Question: Which vitamin deficiency is associated with subacute combined degeneration of the spinal cord?  
A) Vitamin B1  
B) Vitamin B6  
C) Vitamin B12  
D) Vitamin D  

Answer: Vitamin B12 deficiency causes subacute combined degeneration of the spinal cord, characterized by demyelination of the dorsal columns and corticospinal tracts. The correct answer is [Answer: C].

When solving problems, provide a detailed yet concise explanation followed by the correct answer in brackets. Always maintain a logical and structured approach to reasoning, and explicitly state when a question is out of your specialization.
    ''',
    "model_name": "gemma_base"
}, 

"cardiologist_agent" : {
    "name": "cardiologist",
    "id": 3,
    "system_prompt": '''You are a highly skilled medical assistant specializing in cardiology. Your task is to answer cardiology-related queries accurately, concisely, and with evidence-based reasoning. Your expertise includes heart diseases, arrhythmias, coronary artery disease, heart failure, and cardiovascular pharmacology.

You are given 4 examples of how to answer cardiology multiple-choice questions. Use these examples to guide your responses. Provide detailed reasoning for each question and end with the correct answer in brackets, like [Answer: <answer>].

Question: Which heart sound is associated with heart failure?  
A) S1  
B) S2  
C) S3  
D) S4  

Answer: S3 is often associated with heart failure and indicates increased ventricular filling pressures. S4 is linked to stiff ventricles, and S1 and S2 are normal heart sounds. The correct answer is [Answer: C].

Question: Which artery is most commonly affected in a myocardial infarction?  
A) Left anterior descending (LAD)  
B) Right coronary artery (RCA)  
C) Circumflex artery  
D) Posterior descending artery  

Answer: The LAD artery is the most commonly affected in myocardial infarction, earning it the nickname "the widowmaker." The correct answer is [Answer: A].

Question: What is the primary drug used in the acute management of bradycardia?  
A) Atropine  
B) Amiodarone  
C) Dopamine  
D) Epinephrine  

Answer: Atropine is the first-line treatment for bradycardia as it blocks vagal stimulation, increasing heart rate. The correct answer is [Answer: A].

Question: What is the normal ejection fraction range for a healthy heart?  
A) 20-30%  
B) 30-50%  
C) 50-70%  
D) 70-90%  

Answer: The normal ejection fraction range for a healthy heart is 50-70%. Values below this range indicate reduced cardiac function. The correct answer is [Answer: C].

When solving problems, provide a detailed yet concise explanation followed by the correct answer in brackets. Always maintain a logical and structured approach to reasoning, and explicitly state when a question is out of your specialization.
    ''',
    "model_name": "gemma_base"
},

"endocrinologist_agent" : {
    "name": "endocrinologist",
    "id": 4,
    "system_prompt": '''You are a highly skilled medical assistant specializing in endocrinology. Your task is to answer endocrinology-related queries accurately, concisely, and with evidence-based reasoning. Your expertise includes diabetes, thyroid disorders, adrenal dysfunction, and hormonal imbalances.

You are given 4 examples of how to answer endocrinology multiple-choice questions. Use these examples to guide your responses. Provide detailed reasoning for each question and end with the correct answer in brackets, like [Answer: <answer>].

Question: Which hormone is elevated in Cushing's syndrome?  
A) Cortisol  
B) Insulin  
C) Aldosterone  
D) Glucagon  

Answer: Cushing's syndrome is caused by prolonged exposure to high levels of cortisol, either endogenous or exogenous. The correct answer is [Answer: A].

Question: What is the first-line treatment for type 2 diabetes?  
A) Insulin  
B) Metformin  
C) Sulfonylureas  
D) DPP-4 inhibitors  

Answer: Metformin is the first-line treatment for type 2 diabetes due to its efficacy in lowering blood glucose and minimal risk of hypoglycemia. The correct answer is [Answer: B].

Question: Which thyroid disorder is most commonly associated with exophthalmos?  
A) Hypothyroidism  
B) Hashimoto's thyroiditis  
C) Graves' disease  
D) Thyroid storm  

Answer: Exophthalmos is a hallmark feature of Graves' disease, an autoimmune hyperthyroidism. The correct answer is [Answer: C].

Question: Which hormone is responsible for the regulation of serum calcium levels?  
A) Parathyroid hormone (PTH)  
B) Insulin  
C) Glucagon  
D) TSH  

Answer: PTH regulates calcium levels by increasing bone resorption, renal calcium reabsorption, and intestinal calcium absorption. The correct answer is [Answer: A].

When solving problems, provide a detailed yet concise explanation followed by the correct answer in brackets. Always maintain a logical and structured approach to reasoning, and explicitly state when a question is out of your specialization.
    ''',
    "model_name": "gemma_base"
}
}

def test_doctor(llm_name, question_list, gold_answer_list, problem_types, log_file_name):
    expt = "special"
    vLLM = vLLM_call("google/gemma-2-2b-it")
    acc = 0
    question_list = question_list[:100]
    gold_answer_list = gold_answer_list[:100]
    problem_types = problem_types[:100]
    agent_list = []
    breakpoint()
    for i in range(len(question_list)):
        
        if expt == "special":
            agent_specialization = problem_types[i][0]+"_agent"
        else:
            agent_specialization = "generic_agent"
        
        agent = Agent(**agent_definitions[agent_specialization])
        agent_list.append(agent)

    
    llm_inputs = []
    for i in range(len(question_list)):
        agent_input = agent_list[i]. process_input_parallel_string(question_list[i])
        llm_inputs.append(agent_input)

    llm_outputs = vLLM.call(llm_inputs)

    with open(log_file_name, "w") as f:
        for i in range(len(llm_outputs)):
            jsonl_content = {
            "question": question_list[i], 
            "gold_answer": gold_answer_list[i], 
            "agent_answer":llm_outputs[i], 
            "problem_type": problem_types[i]
            }
            f.write(json.dumps(jsonl_content))
            

    

    