# from mathematicians import *
from hospital import *
# from llm import *
# from society import *
from MedQA_single_agent import *
import os
import json
import yaml
import argparse


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    llm_name = config["llm_name"]
    dataset_path = config["dataset_path"]
    log_file_name = config["log_file_name"]

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    question_list = dataset["question_list"]
    gold_answer_list = dataset["gold_answer_list"]
    problem_type_list = dataset["problem_type_list"]
    # test_mathematicians(llm_name, question_list, gold_answer_list, problem_type_list, log_file_name)
    test_doctor(llm_name, question_list, gold_answer_list, problem_type_list, log_file_name)



