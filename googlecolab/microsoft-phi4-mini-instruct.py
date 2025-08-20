from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, "fangcaotank/task-13-microsoft-Phi-4-mini-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
tokenizer.pad_token = tokenizer.eos_token

def ask(prompt):
    inp = f"<|user|>{prompt}<|assistant|>"
    inputs = tokenizer(inp, return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    txt = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return txt.split("<|assistant|>")[-1].strip()

print(ask("Peux-tu extraire le contenu d'un email ?"))
