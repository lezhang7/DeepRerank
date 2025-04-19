import time
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
def get_agent(model_name, api_key=None):
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        agent = QwenClient(model_name=model_name, temperature=0)
    else:
        if api_key is None:
            raise "Please provide OpenAI Key."
        agent = OpenaiClient(api_key)
    return agent

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
    
    def batch_chat(self, messages_batch, return_text=True, **kwargs):
        """Process multiple message sets in a batch to improve throughput."""
        results = []
        
        for messages in messages_batch:
            try:
                result = self.chat(messages=messages, return_text=return_text, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error in batch processing: {str(e)}")
                # Add a placeholder for failed requests
                results.append("ERROR::batch_processing_failed" if return_text else {
                    "choices": [{"message": {"content": "ERROR::batch_processing_failed"}}],
                    "model": self.model_name
                })
        
        return results

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
        # Initialize the model in eval mode to reduce memory usage
        # Get number of GPUs from environment variable CUDA_VISIBLE_DEVICES
        gpu_count = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) if os.environ.get('CUDA_VISIBLE_DEVICES') else 1
        print(f"Using {gpu_count} GPUs")
        self.llm = LLM(model=model_name, tensor_parallel_size=gpu_count)
        # Ensure the model is in evaluation mode

    @torch.no_grad()  # Use decorator to disable gradient tracking
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
                
                # Generate output (no need for with torch.no_grad() due to decorator)
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

    def batch_chat(self, messages_batch, return_text=True, **kwargs):
        """
        Process multiple message sets in a batch using vLLM's efficient batching.
        This can significantly improve throughput compared to sequential processing.
        """
        # Prepare inputs for all messages in the batch
        texts = []
        for messages in messages_batch:
            # Apply chat template to format messages
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
        
        # Update sampling parameters if provided in kwargs
        sampling_params = self.sampling_params
        if kwargs:
            sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', self.sampling_params.temperature),
                top_p=kwargs.get('top_p', self.sampling_params.top_p),
                repetition_penalty=kwargs.get('repetition_penalty', self.sampling_params.repetition_penalty),
                max_tokens=kwargs.get('max_tokens', self.sampling_params.max_tokens)
            )
        
        try:
            # Generate outputs for the entire batch at once
            outputs = self.llm.generate(texts, sampling_params)
            
            # Extract completions
            completions = [output.outputs[0].text for output in outputs]
            
            if return_text:
                return completions
            else:
                # Create a structure similar to OpenAI's response format for each completion
                return [
                    {
                        "choices": [{"message": {"content": completion}}],
                        "model": self.model_name
                    }
                    for completion in completions
                ]
        except Exception as e:
            print(f"Batch processing error: {str(e)}")
            # Return error messages for all requests in the batch
            error_msg = 'ERROR::batch_failed'
            if "context length" in str(e).lower():
                error_msg = 'ERROR::reduce_length'
            
            if return_text:
                return [error_msg] * len(messages_batch)
            else:
                return [
                    {
                        "choices": [{"message": {"content": error_msg}}],
                        "model": self.model_name
                    }
                    for _ in messages_batch
                ]