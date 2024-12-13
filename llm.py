# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it",  device_map = "auto", torch_dtype=torch.float16)
    
def gemma_base(messages):
    # messages are of the form [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}]
    # breakpoint()
    input_text = ""
    for message in messages:
        input_text += f"{message['role']}: {message['content']}\n"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=250,  temperature=0.7, top_p=0.95, top_k=40)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

def LLM(input_text_list, model_name):
    if model_name == "gemma_base":
        return gemma_base(input_text_list)
    else:
        raise ValueError(f"Model {model_name} not found")


