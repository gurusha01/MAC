# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from openai import OpenAI

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
tokenizer = "dummy"
model = "dummy"
checkpoint = "dmis-lab/meerkat-7b-v1.0"
checkpoint = "FreedomIntelligence/HuatuoGPT-o1-7B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint,  device_map = "auto", torch_dtype=torch.float16)
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it",  device_map = "auto", torch_dtype=torch.float16)
    
def gemma_base(messages):
    # messages are of the form [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}]
    # breakpoint()
    input_text = ""
    for message in messages:
        input_text += f"{message['role']}: {message['content']}\n"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1000,  temperature=0.7, top_p=0.95, top_k=40)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    final_answer =  tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    # stop when you see [Answer:<number>] and output everything before that 
    # find the index of [Answer: and output everything till 10 characters after that
    index = final_answer.find("[Answer:")
    if index != -1:
        final_answer = final_answer[:index + 18]
    return final_answer
   
def meerkat(messages):

   
    input_text = ""
    for message in messages:
        input_text += f"{message['role']}: {message['content']}\n"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1000,  temperature=0.7, top_p=0.95, top_k=40)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    final_answer =  tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    # stop when you see [Answer:<number>] and output everything before that 
    # find the index of [Answer: and output everything till 10 characters after that
    index = final_answer.find("[Answer:")
    if index != -1:
        final_answer = final_answer[:index + 18]
    return final_answer
 
    
    


def gpt(messages):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content

def LLM(input_text_list, model_name):
    if model_name == "gemma_base":
        return gemma_base(input_text_list)
    elif model_name == "gpt":
        return gpt(input_text_list)
    elif model_name == "meerkat":    
        return meerkat(input_text_list)
    else:
        raise ValueError(f"Model {model_name} not found")


