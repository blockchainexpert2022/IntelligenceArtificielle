from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Chemin local où vous avez sauvegardé le modèle
model_dir = "./llm-finetuned"

# Charge le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model     = AutoModelForCausalLM.from_pretrained(model_dir)

# Crée le pipeline de génération
# device = 0 → GPU ; device = -1 → CPU
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# Exemple d’utilisation
prompt = "Le futur de l'intelligence artificielle est"
outputs = generator(prompt, max_new_tokens=50, num_return_sequences=1)

print(outputs[0]["generated_text"])
