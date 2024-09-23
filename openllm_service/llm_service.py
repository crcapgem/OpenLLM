import bentoml
from bentoml.io import JSON
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

## Serving: 
# cd openllm_service/
# bentoml serve llm_service.py:svc --reload --timeout 300

# Load the model and tokenizer
model_name = "distilgpt2" #"gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the pad token ID to avoid warnings
tokenizer.pad_token_id = tokenizer.eos_token_id

# Create a BentoML service
svc = bentoml.Service("llm_text_generation_service")

@svc.api(input=JSON(), output=JSON())
def generate_text(input_data: dict) -> dict:
    prompt = input_data.get("prompt", "")
    
    # Tokenize the input prompt, padding to the longest sequence
    inputs = tokenizer( 
        prompt,
        return_tensors="pt",
        padding=True,  # Pads the input to the longest sequence length
        truncation=True,
        max_length=1024,  # Ensure input length is within the model limit
    )

    # Generate text using the model, using the attention mask
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=1024,  # Set the max length of generated text
            num_return_sequences=1,
            do_sample=True,
            temperature=0.1  # Adjust temperature for creativity
        )

    # Decode the generated tokens into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_text": generated_text}
