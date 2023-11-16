import sys
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List
import numpy as np
from time import perf_counter
import logging
import torch
import GPUtil


# Set up logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)


class PolyglotGenerator:
    """Class with only class methods"""

    # Class variable for the model pipeline
    loaded_tokenizer = None
    loaded_model = None

    @classmethod
    def load(polyglot_model):
        # Only load one instance of the model
        if polyglot_model.loaded_model is None:
            # Load the model parameters in the memory
            # Note: device_map="auto" allows us to use multiple GPU
            t0 = perf_counter()
            model_name= "EleutherAI/polyglot-ko-12.8b"
            polyglot_model.loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.bfloat16
            polyglot_model.loaded_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=dtype, load_in_8bit=True)
            polyglot_model.loaded_model.eval()
            elapsed = 1000 * (perf_counter() - t0)
            log.info("Model warm-up time: %d ms.", elapsed)

    @classmethod
    def generate_text(polyglot_model, prompt, max_new_tokens=300, temperature=0.9, top_p=0.97, top_k=20, repetition_penalty=1.07):
        # Make sure the model is loaded 
        polyglot_model.load()

        t0 = perf_counter()
        # Encode the input text
        input_ids = polyglot_model.loaded_tokenizer.encode(prompt, return_tensors="pt").to("cuda")

        # Generate text using the model
        output = polyglot_model.loaded_model.generate(
            input_ids = input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=6,
            bad_words_ids=[[202],[63],],
            num_return_sequences=1,  # You can adjust the number of generated sequences as needed
        )

        # Decode and return the generated text
        generated_text = polyglot_model.loaded_tokenizer.decode(output[0], skip_special_tokens=True)

        elapsed = (perf_counter() - t0)
        total_tokens = len(output[0])
        tokens_per_second = total_tokens/elapsed
        log.info("Time for inference: %.2f sec total, %.2f tokens/sec", elapsed, tokens_per_second)

        GPUs = GPUtil.getGPUs()
        total_memory_usage = 0
        for gpu in GPUs:
            log.info(f"GPU %s: %s, Memory Usage: %.2f MB", gpu.id, gpu.name, gpu.memoryUsed)
            total_memory_usage += gpu.memoryUsed
        log.info("Total Memory usage across all GPUs: %.2f", total_memory_usage)

        return generated_text


