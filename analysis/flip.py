# see when an answer is flipped - correct to incorrect and vice versa
# first take input as a model, round, architecture and expt. 
# now open the files phase2_expts/math/{expt}_agents_{round-1}/MAC_T0_{model}_{architecture}.jsonl 
# and phase2_expts/math/{expt}_agents_{round}/MAC_T0_{model}_{architecture}.jsonl
# for each question, extract the "gold_answer" and "math1" for file 1 and 2
# extract text between "Your Openion" and "math2's openion" for file 1 and 2
# see if the answer is different in the two files
# if the answer goes from correct to incorrect or vice versa, print the question and the two answers
# 

from gettext import find
import json
import math
import sys
import os
import re
import torch

def extract_answer(text):
    try:
        answer = text.split("[Answer:")[1].split("]")[0]
        return answer
    except:
        return "None"

def flip(model, round, architecture, expt, agent):
    file1 = f"phase2_expts/math/{expt}_agents_{round-1}_round/MAC_T0_{model}_{architecture}.jsonl"
    file2 = f"phase2_expts/math/{expt}_agents_{round}_round/MAC_T0_{model}_{architecture}.jsonl"
    with open(file1, 'r') as f:
        data1 = f.readlines()
    with open(file2, 'r') as f:
        data2 = f.readlines()
    for i in range(len(data1)):
        # breakpoint()
        c1 = json.loads(data1[i])
        c2 = json.loads(data2[i])
        gold_answer = c1['gold_answer']
        math1_memory1 = c1[agent]
        math1_memory2 = c2[agent]

        # search fro the text between "Your Openion" and "math2's openion" ans return the last such text
        math1_answer1 = re.findall(r'Your Opinion(.*?) Opinion', math1_memory1, re.DOTALL)[-1]
        math1_answer2 = re.findall(r'Your Opinion(.*?) Opinion', math1_memory2, re.DOTALL)[-1]

        math1_a1 = extract_answer(math1_answer1).strip()
        math1_a2 = extract_answer(math1_answer2).strip()
        
        if math1_a1 != math1_a2 and (math1_a1 == gold_answer or math1_a2 == gold_answer):
            # positive flip if the answer goes from correct to incorrect and negative flip if the answer goes from incorrect to correct so positive flip is a bad thing.
            flip_ans = "positive" if (math1_a1 == gold_answer) else "negative"
            # make a file with the name of the model, round, architecture and expt and flip and write the question gold_ans and memory of two llms in a jsonl format
            with open(f"analysis/logs/{model}_{round}_{architecture}_{expt}_{flip_ans}.jsonl", 'a') as f:
                content = {
                    "question": c1['question'],
                    "gold_answer": gold_answer,
                    "answer_before": math1_answer1,
                    "answer_after": math1_answer2,
                    "context_before": math1_memory1,
                    "context_after": math1_memory2, 
                }
                f.write(json.dumps(content) + "\n")
                
# write a function to look through all positive and negative files, extract answer_before and answer_after and compare their lengths and print the average lengths of the two per file 
def compare_lengths(dir_name):
    files = os.listdir(dir_name)
    
    for file in files:
        total_before = 0
        total_after = 0
        with open(f"{dir_name}/{file}", 'r') as f:
            data = f.readlines()
        for line in data:
            content = json.loads(line)
            # breakpoint()
            total_before += len(content['answer_before'])
            total_after += len(content['answer_after'])
        print(f"File: {file}")
        print(f"Average length of answer_before: {total_before/len(data)}")
        print(f"Average length of answer_after: {total_after/len(data)}")
        print("\n")
    
# for each file in analysis/logs, extract the context_before and extract all the [Answer:*] in the context and see how many times is the corect answer given ans how many times incorrect and print their order
def extract_answer_order(dir_name):
    files = os.listdir(dir_name)
    # breakpoint()
    for file in files:
        if "special" in file:
            with open(f"{dir_name}/{file}", 'r') as f:
                data = f.readlines()
            for line in data:
                content = json.loads(line)
                # breakpoint()
                answers = re.findall(r'\[Answer:(.*?)\]', content['context_before'])
                result_string = ""
                for i in range(len(answers)):
                    if answers[i].strip() == content['gold_answer']:
                        result_string += "C"
                    else:
                        result_string += "I"

                print(f"File: {file}")
                print(f"Answer order: {result_string}")
                if(len(result_string)>0):
                        print("------------------------------------------------------------------------------------------------------------------------")
                    # if result_string[-1]=="C" and "negative" in file:
                        print("question: ", content['question'])
                        print("gold_answer: ", content['gold_answer'])
                        print("**********************************************************")
                        print("answer_before: ", content['answer_before'])
                        print("**********************************************************")
                        print("answer_after: ", content['answer_after'])
                        print("**********************************************************")
                        print("history_before: ", content['context_before'])    
                    # if result_string[-1]=="I" and "positive" in file:
                        # print("question: ", content['question'])
                print("\n")
                breakpoint()

def find_perplexity(text, model, tokenizer):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)

    # Shift inputs for causal language modeling
    # We want to predict each token given all previous tokens
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Cross-entropy loss

    # Calculate perplexity
    perplexity = torch.exp(loss)
    print(f"Perplexity: {perplexity.item()}")
    return perplexity.item()
    
    

def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load Qwen-32B tokenizer and model
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Replace with your model name or path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    # open file in analysis/logs and extract the answer_before and answer_after find the perplexity of the two
    with open("analysis/logs/qwen32_2_ring_generic_negative.jsonl", 'r') as f:
        data = f.readlines()
    for line in data:
        content = json.loads(line)
        text1 = f"Question: {content['question']} \n \n Answer: Let's think step by step. {content['answer_before']}"
        
        pp1 = find_perplexity(text1, model, tokenizer)
        text2 = f"Question: {content['question']} \n {content['context_before']} \n Answer: Let's think step by step. {content['answer_after']}"
        pp2 = find_perplexity(text2, model, tokenizer)
        if pp1 < pp2:
            print("****************************************************************************************************")
            print("Question: ", content['question'])
            print("Gold Answer: ", content['gold_answer'])
            print("Answer before: ", content['answer_before'])
            print("Answer after: ", content['answer_after'])
            print("****************************************************************************************************")

        print("****************************************************************************************************")
        # breakpoint()

    
    
    
    
    
    # find_perplexity("What is the capital of France?", model, tokenizer)

if __name__ == "__main__":
    # take arcuments from the command line
    
    # exit()
    load_model()
    # extract_answer_order("analysis/logs")
    # model = sys.argv[1]
    # expt = sys.argv[2]
    # analyse = sys.argv[3]
    # rounds = [2,]
    # agents = ["math1", "math2", "math3", "math4", "math5"]
    # agents = ["geometry", "number_theory", "algebra", "calculus", "permutation"]
    # architectures = ["heirarchical", "ring", "clique"]
    # if analyse=="1":
    #     for round in rounds:
    #         for architecture in architectures:
    #             for agent in agents:
    #                 flip(model, round, architecture, expt, agent)
    # else:
    #     compare_lengths("analysis/logs")

    # # example how to call from command line
    # # python analysis/flip.py gpt2 2 gpt2 math


