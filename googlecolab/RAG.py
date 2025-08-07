import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- ÉTAPE DE PRÉPARATION (à faire une seule fois) ---

# 1. Charger notre source de connaissances
with open("projet_secret.txt", "r", encoding="utf-8") as f:
    # On divise le document en "chunks" (ici, par ligne)
    knowledge_chunks = [line.strip() for line in f.readlines() if line.strip()]

# 2. Charger un modèle pour créer des "embeddings" (représentations numériques du sens)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Créer les embeddings pour chaque chunk de notre document
#    C'est notre "base de données vectorielle" simplifiée
knowledge_embeddings = embedding_model.encode(knowledge_chunks)

print("✅ Base de connaissances prête et indexée.\n")


# --- ÉTAPE D'INFERENCE RAG (à chaque question de l'utilisateur) ---

# La question de l'utilisateur
user_question = "Quel est le budget du projet Albatros et qui le dirige ?"

# 1. RETRIEVAL (Récupération)
# On transforme la question en embedding
question_embedding = embedding_model.encode([user_question])

# On calcule la similarité entre la question et chaque chunk du document
similarities = cosine_similarity(question_embedding, knowledge_embeddings)

# On trouve le chunk le plus pertinent (celui avec le score le plus haut)
most_relevant_chunk_index = np.argmax(similarities)
context = knowledge_chunks[most_relevant_chunk_index]

# Pour un vrai RAG, on prendrait les 'k' meilleurs chunks
# Ici on prend les 2 meilleurs pour un contexte plus riche
top_2_indices = np.argsort(similarities[0])[-2:]
context = "\n".join([knowledge_chunks[i] for i in top_2_indices])

print(f"🔍 Contexte pertinent trouvé dans les documents :\n---\n{context}\n---\n")

# 2. AUGMENTED GENERATION (Génération Augmentée)
# On crée un prompt qui inclut le contexte et la question
prompt_template = f"""
En te basant UNIQUEMENT sur le contexte suivant, réponds à la question.

Contexte:
{context}

Question:
{user_question}
"""

print("💬 Prompt final envoyé au LLM :\n---\n" + prompt_template + "\n---\n")

# On initialise le client Hugging Face
client = InferenceClient()

# On envoie le prompt enrichi au LLM
completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3-0324",
    messages=[
        {"role": "user", "content": prompt_template}
    ],
)

# On affiche la réponse, qui sera maintenant factuellement correcte !
print("🤖 Réponse du LLM (basée sur le RAG) :\n")
print(completion.choices[0].message.content)
