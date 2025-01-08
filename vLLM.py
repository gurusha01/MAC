
import os 
import torch
from vllm import LLM, SamplingParams

class vLLM_call:
    def __init__(self, model):
        self.model = model
        self.llm = LLM(model=self.model, 
                    #    tensor_parallel_size = 4
                       )
    
    def call(self, prompts):
        # breakpoint()
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        response_list = []
        outputs =  self.llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            response_list.append(generated_text)
        return response_list
    