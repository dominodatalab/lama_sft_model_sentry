import mlflow
import transformers

class MyModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import os
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig
        )

        self.project_id = os.listdir('/artifacts/mlflow')[0]
        compute_dtype = getattr(torch, "float16")
        quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=compute_dtype,
                                          bnb_4bit_use_double_quant=False)

        ft_model_name = "final_merged_checkpoint"
        model_tokenizer_path = f"/artifacts/mlflow/{self.project_id}/{ft_model_name}"
        
        self.model = AutoModelForCausalLM.from_pretrained(model_tokenizer_path,
                                                          cache_dir=f"/artifacts/mlflow/{self.project_id}/cache/",
                                                          quantization_config=quant_config,
                                                          device_map="auto")
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path, 
                                                       cache_dir=f"/artifacts/mlflow/{self.project_id}/cache/",
                                                       trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    
    
    def predict(self, context, model_input, params=None):
        """
        This method generates prediction for the given input.
        """
        prompt = model_input["prompt"]

        if prompt is None:
            return 'Please provide a prompt.'
        
        prompt_template = f"<s>[INST] {{dialogue}} [/INST]"
            

        user_input = f"<s>[INST] {prompt} [/INST]"
        
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(user_input))
        input_length = len(tokens)
        
        new_tokens = 750
        
        text = f"<s>[INST] {prompt} [/INST]"

        device = "cuda:0"

        inputs = self.tokenizer(text, return_tensors="pt").to(device)

        generation_config = transformers.GenerationConfig(
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens = 200
                )

        outputs = self.model.generate(**inputs, generation_config=generation_config)
        llm_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = llm_output.replace(f"[INST] {prompt} [/INST]", '')
        return {'text_from_llm': result}
