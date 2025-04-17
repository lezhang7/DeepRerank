import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class OpenaiClient:
    def __init__(self, keys=None, start_id=None, proxy=None):
        from openai import OpenAI
        import openai
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        self.api_key = self.key[self.key_id % len(self.key)]
        self.client = OpenAI(api_key=self.api_key)

    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(*args, **kwargs, timeout=30)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].message.content
        return completion

    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.completions.create(
                    *args, **kwargs
                )
                break
            except Exception as e:
                print(e)
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].text
        return completion
    


class QwenClient:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", temperature=0.7, top_p=0.8, 
                repetition_penalty=1.05, max_tokens=2048):
        self.model_name = model_name
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set default sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens
        )
        # Initialize the model
        self.llm = LLM(model=model_name, tensor_parallel_size=1)

    def chat(self, messages, return_text=True, **kwargs):
        while True:
            try:
                # Apply chat template to format messages
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Update sampling parameters if provided in kwargs
                sampling_params = self.sampling_params
                if kwargs:
                    sampling_params = SamplingParams(
                        temperature=kwargs.get('temperature', self.sampling_params.temperature),
                        top_p=kwargs.get('top_p', self.sampling_params.top_p),
                        repetition_penalty=kwargs.get('repetition_penalty', self.sampling_params.repetition_penalty),
                        max_tokens=kwargs.get('max_tokens', self.sampling_params.max_tokens)
                    )
                
                # Generate output
                outputs = self.llm.generate([text], sampling_params)
                completion = outputs[0].outputs[0].text
                break
            except Exception as e:
                print(str(e))
                if "context length" in str(e).lower():
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        
        if return_text:
            return completion
        else:
            # Create a structure similar to OpenAI's response format
            return {
                "choices": [{"message": {"content": completion}}],
                "model": self.model_name
            }

    def text(self, prompt, return_text=True, **kwargs):
        # For text completion, we'll create a simple user message
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, return_text=return_text, **kwargs)