!pip install transformers torch bitsandbytes accelerate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model you want to use
model_id = "google/gemma-2b-it"#"deepseek-ai/DeepSeek-V3-0324"

# Load the tokenizer and model from the local cache (or download if not present)
# device_map="auto" will automatically use a GPU if available
# torch_dtype="auto" will select the best data type for your hardware
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

# Define the messages in the chat format the model expects
messages = [
    {"role": "user", "content": "What date is it today ?"}
]

# Use the tokenizer to apply the chat template and convert to tensor format
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device) # Move the input tensor to the same device as the model

# Generate a response
# The model will output a sequence of token IDs
outputs = model.generate(
    input_ids,
    max_new_tokens=512,  # You can adjust the maximum length of the response
    eos_token_id=tokenizer.eos_token_id
)

# Decode the generated token IDs back into a string
response_ids = outputs[0][input_ids.shape[-1]:]
response = tokenizer.decode(response_ids, skip_special_tokens=True)

# Print the response
print(response)
