# ==============================================================================
#      CODE COMPLET AVEC VÉRIFICATION DE L'INSTALLATION
# ==============================================================================

# --- ÉTAPE 1 : Installation conditionnelle des bibliothèques ---
try:
    # On vérifie la bibliothèque principale
    import transformers
    print("✅ Les bibliothèques nécessaires sont déjà installées.")
except ImportError:
    print("⏳ Installation des bibliothèques (transformers, torch)...")
    !pip install transformers torch -q
    print("✅ Installation terminée.")

# --- Le reste de votre code reste identique ---
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Détecter le GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du périphérique : {device}")

# Charger le modèle et le tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
print(f"Modèle '{model_name}' chargé sur le {device}.")

# Préparer et générer le texte
prompt_text = "Quelles sont les bases de la programmation Windev en W-Language ?"
input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

output_sequences = model.generate(
    input_ids,
    max_length=80,
    num_return_sequences=1
)

generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print("\n--- RÉSULTAT ---")
print(generated_text)
