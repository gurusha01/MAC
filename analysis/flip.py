# see when an answer is flipped - correct to incorrect and vice versa
# first take input as a model, round, architecture and expt. 
# now open the files phase2_expts/math/{expt}_agents_{round-1}/MAC_T0_{model}_{architecture}.jsonl 
# and phase2_expts/math/{expt}_agents_{round}/MAC_T0_{model}_{architecture}.jsonl
# for each question, extract the "gold_answer" and "math1" for file 1 and 2
# extract text between "Your Openion" and "math2's openion" for file 1 and 2
# see if the answer is different in the two files
# if the answer goes from correct to incorrect or vice versa, print the question and the two answers
# 

import json
import math
import sys
import os
import re

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
    



if __name__ == "__main__":
    # take arcuments from the command line
    
    # exit()
    
    model = sys.argv[1]
    expt = sys.argv[2]
    analyse = sys.argv[3]
    rounds = [2,]
    agents = ["math1", "math2", "math3", "math4", "math5"]
    architectures = ["heirarchical", "ring", "clique"]
    if analyse=="1":
        for round in rounds:
            for architecture in architectures:
                for agent in agents:
                    flip(model, round, architecture, expt, agent)
    else:
        compare_lengths("analysis/logs")

    # example how to call from command line
    # python analysis/flip.py gpt2 2 gpt2 math


