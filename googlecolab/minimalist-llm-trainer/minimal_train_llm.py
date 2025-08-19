# ---------------------------------------------------------
# train_llm.py – fine‑tuning d'un petit LLM avec 🤗 Transformers
# ---------------------------------------------------------

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling, # Import DataCollatorForLanguageModeling
)

# 1️⃣ Jeu de données très petit (ici, le jeu "wikitext-2" en anglais)
train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")  # 1 % du jeu → < 10 k lignes

# 2️⃣ Tokenizer et modèle (un petit GPT‑2 déjà pré‑entrainé)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Add a pad token if the tokenizer does not have one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3️⃣ Tokenisation + création des « labels » (shiftés d’un token)
def tokenize(example):
    # encode le texte complet, on limite à 128 tokens (vous pouvez changer)
    # Add padding to max length
    tokenized = tokenizer(example["text"], truncation=True, max_length=128, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()   # les cibles sont les mêmes que les entrées
    return tokenized

tokenized_dataset = train_dataset.map(tokenize, remove_columns=["text"])

# Use DataCollatorForLanguageModeling to handle padding and shifting of labels
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# 4️⃣ Arguments d’entraînement ultra‑simples
training_args = TrainingArguments(
    output_dir="./llm-finetuned",   # où sauvegarder le modèle
    per_device_train_batch_size=4,  # taille de batch (ajustez selon votre RAM/VRAM)
    num_train_epochs=1,             # 1 epoch suffit pour un test
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),   # mixed‑precision si GPU
)

# 5️⃣ Trainer (tout est déjà encapsulé)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator, # Add data_collator
)

# 6️⃣ Lancement de l’entraînement
trainer.train()

# 7️⃣ Sauvegarde finale (facultatif, déjà sauvegardé dans output_dir)
trainer.save_model("./llm-finetuned")
tokenizer.save_pretrained("./llm-finetuned")
print("✅ Entraînement terminé, modèle sauvegardé dans ./llm-finetuned")
