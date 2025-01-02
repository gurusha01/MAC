# here we define a society of mathematicians that work together to solve problems from math dataset. 
# First there is a decomposer that decomposes the problem into smaller subproblems and distributes them to the mathematicians. 
# Then there are mathemaricians that specialize in different feilds like algebra, calculus, geometry, etc. 
# they work with other math researchers that are further specialized in different areas like number area calculation, counting principal,etc.
# after conversation, they report their answers to an aggregator that aggregates the answers and returns the final answer.
# Currently we start with MATH dataset, then we include HardMAth, Scibench etc as well. 

'''
Instantiate agents
make few graphs
load datasets
evaluate those graphs on dataset
'''
from society import Agent, Society
from llm import *
import json
from tqdm import tqdm


# provide 4 shots of examples of problem solving to each agent related to their topic.
gyneacologist_agent = {
    "name": "gyneacologist",
    "id": 1,
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
}

oncologist_agent = {
    "name": "oncologist",
    "id": 3,
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
}

neurologist_agent = {
    "name": "neurologist",
    "id": 4,
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
}

cardiologist_agent = {
    "name": "cardiologist",
    "id": 5,
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
}

endocrinologist_agent = {
    "name": "endocrinologist",
    "id": 6,
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

aggregator_agent = {
    "name": "aggregator",
    "id": 7,
    "system_prompt": '''You are an aggregator agent tasked with synthesizing and evaluating responses from specialized medical agents. Your goal is to carefully analyze their responses along with the original question and provide the most accurate, evidence-based, and well-justified answer.

When responding:
1. Review the responses from the specialized agents and assess their relevance, accuracy, and reasoning. 
2. Consider the complexity of the question and the domains involved. If multiple specializations are relevant, integrate their insights for a holistic response.
3. Resolve discrepancies or conflicts in the responses by referring to established medical guidelines or evidence-based practices.
4. Present the final answer clearly and concisely, with logical reasoning and justification for your decision.

Format the final answer:
- Provide a short explanation for your reasoning.
- Conclude with the answer in brackets like [Answer: <answer>] where <answer> is a choice (A/B/C/D) or a specific term/diagnosis based on the question.

If the question does not fall within any medical domain or is not sufficiently clear, state that it is outside the scope of the aggregated expertise and request clarification or more information.
    ''',
    "model_name": "gemma_base"
}

def test_doctors(llm_name, question_list, gold_answer_list, problem_types, log_file_name):
    acc = 0
    question_list = question_list[19:100]
    gold_answer_list = gold_answer_list[19:100]
    problem_types = problem_types[19:100]
    for i in tqdm(range(len(question_list))):
        
        question = question_list[i]
        gold_answer = gold_answer_list[i]
        problem_type = problem_types[i]
        gyneacologist = Agent(**gyneacologist_agent)
        oncologist = Agent(**oncologist_agent)
        neurologist = Agent(**neurologist_agent)
        cardiologist = Agent(**cardiologist_agent)
        endocrinologist = Agent(**endocrinologist_agent)
        aggregator = Agent(**aggregator_agent)


        doctors = [gyneacologist, oncologist, neurologist, cardiologist, endocrinologist, aggregator]

        heirarchical_hospital_graph = {
            "aggregator": [],
            "gyneacologist": [aggregator],
            "oncologist": [aggregator],
            "neurologist": [aggregator],
            "cardiologist": [aggregator],
            "endocrinologist": [aggregator],
        }

        clique_hospital_graph = {
            "aggregator": [],
            "gyneacologist": [oncologist, neurologist, cardiologist, endocrinologist, aggregator],
            "oncologist": [gyneacologist, neurologist, cardiologist, endocrinologist, aggregator],
            "neurologist": [gyneacologist, oncologist, cardiologist, endocrinologist, aggregator],
            "cardiologist": [gyneacologist, oncologist, neurologist, endocrinologist, aggregator],
            "endocrinologist": [gyneacologist, oncologist, neurologist, cardiologist, aggregator],
        }
        
        if "gyneacologist" in problem_type:
            problem_specific_agent = gyneacologist
        elif "oncologist" in problem_type:
            problem_specific_agent = oncologist
        elif "neurologist" in problem_type:
            problem_specific_agent = neurologist
        elif "cardiologist" in problem_type:
            problem_specific_agent = cardiologist
        elif "endocrinologist" in problem_type:
            problem_specific_agent = endocrinologist
        elif "headdoctor" in problem_type:
            problem_specific_agent = aggregator
        
        single_doctor = [problem_specific_agent, aggregator]
        single_doctor_graph = {
            "aggregator": [],
            "gyneacologist": [aggregator],
            "oncologist": [aggregator],
            "neurologist": [aggregator],
            "cardiologist": [aggregator],
            "endocrinologist": [aggregator],
        }

        # instantiate the society
        # ProblemSolver = Society(doctors, heirarchical_hospital_graph)
        # ProblemSolver = Society(doctors, clique_hospital_graph)
        ProblemSolver = Society(single_doctor, single_doctor_graph)
        ProblemSolver.run_simulation(1, question, background = "")

        # based on the memory of head doctor, give the final answer to the user
        # breakpoint()
        # aggregator_response = aggregator.last_utterance + question
        
        # messages = [
        #     {"role": "system", "content": answer_extraction_prompt},
        #     {"role": "user", "content": aggregator_response + "n Answer: Let's think step by step."}
        # ]

        # response = LLM(messages, llm_name)
        response = aggregator.last_utterance
        # print(response)
        try:
            response = response.split("[Answer:")[1].strip()
        except:
            response = response.strip()
        # print(response)
        # print(response, gold_answer)
        if(gold_answer in response):
            acc+=1
        # print(acc)

        # save the question, answer and memory of each agent in a jsonl file
        jsonl_content = {
            "question": question, 
            "gold_answer": gold_answer, 
            "society_answer":response, 
            "problem_type": problem_type,
            "gyneacologist": gyneacologist.memory, 
            "oncologist": oncologist.memory,
            "neurologist": neurologist.memory,
            "cardiologist": cardiologist.memory,
            "endocrinologist": endocrinologist.memory,
            "aggregator": aggregator.memory
        }
        with open(log_file_name, "a") as f:
            f.write(json.dumps(jsonl_content) + "\n")

    print("Accuracy: ", acc/len(question_list))







