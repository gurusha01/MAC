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


from datasets import load_dataset

ds = load_dataset("ai2lumos/lumos_maths_plan_iterative")

# instantiate all the doctors in the hospital
geometry_agent = {
    "name": "geometry",
    "id": 1,
    "system_prompt": '''You are a highly skilled mathematical assistant specializing in geometry. Your task is to solve geometry problems with precision and clarity. You understand concepts such as Euclidean geometry, coordinate geometry, trigonometry, and geometric proofs.

    When solving problems, provide detailed yet concise answers with examples. Always maintain a logical and structured approach to problem-solving. It could be that the problem is not a geometry problem, in that case, you should say that it is out of your domain. Think about the solution and give a brief summary of your solution in <Opinion></Opinion> tags like <Opinion>To solve...</Opinion>  or <Opinion>I think...</Opinion> etc.''',
    "model_name": "gemma_base"
    }

permutation_agent = {
    "name": "permutation",
    "id": 2,
    "system_prompt": '''
    You are a highly skilled mathematical assistant specializing in combinatorics. Your task is to solve problems involving permutations, combinations, counting principles with precision and clarity.

    When solving problems, provide detailed yet concise answers with relevant examples. Always maintain a logical and structured approach to problem-solving. It could be that the problem is not a combinatorics problem, in that case, you should say that it is out of your domain. Think about the solution and give a brief summary of your solution in <Opinion></Opinion> tags like <Opinion>To solve...</Opinion>  or <Opinion>I think...</Opinion> etc.''',
    "model_name": "gemma_base"
    }

number_theory_agent = {
    "name": "number_theory",
    "id": 3,
    "system_prompt": '''You are a highly skilled mathematical assistant specializing in number theory. Your task is to solve problems related to prime numbers, divisibility, modular arithmetic, greatest common divisors, least common multiples, Diophantine equations, and more with precision and clarity.

    When solving problems provide detailed yet concise answers with relevant examples. Always maintain a structured and methodical approach to problem-solving. It could be that the problem is not a number theory problem, in that case, you should say that it is out of your domain. Think about the solution and give a brief summary of your solution in <Opinion></Opinion> tags like <Opinion>To solve...</Opinion>  or <Opinion>I think...</Opinion> etc.''',
    "model_name": "gemma_base"
    }

algebra_agent = {
    "name": "algebra",
    "id": 4,
    "system_prompt": '''You are a highly skilled mathematical assistant specializing in algebra. Your task is to solve problems involving equations, inequalities, polynomials, matrices, and linear algebra with precision and clarity.

    When solving problems provide detailed yet concise answers with relevant examples. Always maintain a logical and systematic approach to problem-solving. It could be that the problem is not a algebra problem, in that case, you should say that it is out of your domain. Think about the solution and give a brief summary of your solution in <Opinion></Opinion> tags like <Opinion>To solve...</Opinion>  or <Opinion>I think...</Opinion> etc.''',
    "model_name": "gemma_base"
    }

calculus_agent = {
    "name": "calculus",
    "id": 5,
    "system_prompt": '''You are a highly skilled mathematical assistant specializing in calculus. Your task is to solve problems involving differentiation, integration, limits, series, and multivariable calculus with precision and clarity.

    When solving problems provide detailed yet concise answers with relevant examples and visual aids if needed. Always maintain a logical and structured approach to problem-solving. It could be that the problem is not a calculus problem, in that case, you should say that it is out of your domain.
    ''',
    "model_name": "gemma_base"
    }

aggregator_agent = {
    "name": "aggregator",
    "id": 6,
    "system_prompt": '''You are an aggregator agent tasked with synthesizing and evaluating responses from specialized mathematical agents (Number Theory Agent, Calculus Agent, Algebra Agent, Geometry Agent, and Combinatorics Agent). Your goal is to carefully analyze their responses along with the original question and provide the most accurate and well-justified answer.

    When processing information:
    1. Read and understand the original question to identify the mathematical domain(s) involved.
    2. Review responses from the specialized agents, evaluating the correctness, clarity, and relevance of their solutions.
    3. If the responses conflict or overlap, resolve inconsistencies by:
    - Identifying the response that directly addresses the problem most accurately.
    - Providing justification for selecting or synthesizing responses.
    4. If necessary, combine insights from multiple agents to form a coherent, accurate, and complete answer.

    Present the final answer:
    1. Clearly and concisely, with logical reasoning and justification for your decision.
    2. Ensure the solution is correct, easy to understand, and formatted using proper mathematical notation.

    If no agent provides a suitable response or more information is needed, state what is missing and suggest next steps. 
    Give the final answer in brackets like [Answer: <answer>] where <answer> is a numeric value like [Answer: 123] or [Answer: 123.45] etc.
    ''',
    "model_name": "gemma_base"
    }

def test_mathematicians(llm_name, question_list, gold_answer_list, log_file_name):
    acc = 0
    question_list = question_list[:100]
    gold_answer_list = gold_answer_list[:100]
    for i in tqdm(range(len(question_list))):
        
        question = question_list[i]
        gold_answer = gold_answer_list[i]

        geometry = Agent(**geometry_agent)
        permutation = Agent(**permutation_agent)
        number_theory = Agent(**number_theory_agent)
        algebra = Agent(**algebra_agent)
        calculus = Agent(**calculus_agent)
        aggregator = Agent(**aggregator_agent)


        mathematicians = [geometry, permutation, number_theory, algebra, calculus, aggregator]

        heirarchical_mathematicians_graph = {
            "aggregator": [],
            "geometry": [aggregator],
            "permutation": [aggregator],
            "number_theory": [aggregator],
            "algebra": [aggregator],
            "calculus": [aggregator],
        }

        # instantiate the society
        ProblemSolver = Society(mathematicians, heirarchical_mathematicians_graph)
        ProblemSolver.run_simulation(1, question, background = "")

        # based on the memory of head doctor, give the final answer to the user
        # breakpoint()
        # aggregator_response = aggregator.last_utterance + question
        
        # messages = [
        #     {"role": "system", "content": answer_extraction_prompt},
        #     {"role": "user", "content": aggregator_response + "\n Answer: Let's think step by step."}
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
            "geometry": geometry.memory, 
            "permutation": permutation.memory, 
            "number_theory": number_theory.memory, 
            "algebra": algebra.memory, 
            "calculus": calculus.memory, 
            "aggregator": aggregator.memory
        }
        with open(log_file_name, "a") as f:
            f.write(json.dumps(jsonl_content) + "\n")

    print("Accuracy: ", acc/len(question_list))







