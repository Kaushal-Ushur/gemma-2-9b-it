import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import huggingface_hub

hf_token = os.getenv("HF_TOKEN")
huggingface_hub.login(hf_token)

class InferlessPythonModel:
    def initialize(self):
        model_name = "google/gemma-2-9b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(model=model_name,enforce_eager=True)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")

    def infer(self, inputs):
        prompt = inputs["prompt"]
        temperature = inputs.get("temperature", 0.7)
        top_p = inputs.get("top_p", 0.1)
        top_k = inputs.get("top_k", 40)
        repetition_penalty = float(inputs.get("repetition_penalty", 1.18))
        max_new_tokens = inputs.get("max_new_tokens", 256)
      
        # messages = [{"role": "user", "content": prompt}]
        # input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

        # input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to("cuda")

        # outputs = self.model.generate(
        #     **input_ids, 
        #     max_new_tokens=max_new_tokens, 
        #     temperature=temperature, 
        #     top_p=top_p, 
        #     top_k=top_k, 
        #     repetition_penalty=repetition_penalty
        # )
        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return {"generated_text":generated_text}

        sampling_params = SamplingParams(
            temperature=temperature,top_p=top_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,max_tokens=max_new_tokens
        )

        result = self.llm.generate([prompt], sampling_params)
        result_output = [output.outputs[0].text for output in result]
        
        return {"generated_text":result_output[0]}

    def finalize(self):
        self.model = None
