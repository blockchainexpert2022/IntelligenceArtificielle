# ==============================================================================
#  CODE COMPLET POUR UTILISER UN MODÈLE HUGGING FACE (GPT-2) DANS GOOGLE COLAB
# ==============================================================================

# ------------------------------------------------------------------------------
#  ÉTAPE 1 : Installation des bibliothèques nécessaires
# ------------------------------------------------------------------------------
# On installe 'transformers' pour accéder aux modèles et 'torch' qui est le
# framework d'apprentissage profond utilisé par de nombreux modèles.
# L'option '-q' signifie 'quiet' pour une installation moins verbeuse.
print("Installation des bibliothèques nécessaires...")
!pip install transformers torch -q

# ------------------------------------------------------------------------------
#  ÉTAPE 2 : Importation des classes requises
# ------------------------------------------------------------------------------
# Nous importons le modèle et son tokenizer.
# - Le Tokenizer prépare le texte pour le modèle (le transforme en nombres).
# - Le Modèle (ici, GPT2LMHeadModel) est l'architecture utilisée pour la génération de texte.
from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("\nBibliothèques importées.")

# ------------------------------------------------------------------------------
#  ÉTAPE 3 : Chargement du modèle et du tokenizer pré-entraînés
# ------------------------------------------------------------------------------
# On spécifie l'identifiant du modèle que l'on souhaite utiliser depuis le Hub Hugging Face.
# Pour cet exemple, nous utilisons 'gpt2', un modèle de génération de texte bien connu.
model_name = 'gpt2'
print(f"\nChargement du tokenizer et du modèle '{model_name}'. Cela peut prendre un moment...")

# Chargement du tokenizer associé au modèle 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Chargement du modèle pré-entraîné 'gpt2' pour la génération de texte (Language Modeling)
model = GPT2LMHeadModel.from_pretrained(model_name)

print("Modèle et tokenizer chargés avec succès.")

# ------------------------------------------------------------------------------
#  ÉTAPE 4 : Préparation de l'input (la phrase de départ)
# ------------------------------------------------------------------------------
# C'est la phrase que vous donnez au modèle pour qu'il la continue.
# N'hésitez pas à la modifier pour tester !
prompt_text = "La technologie de l'intelligence artificielle a beaucoup évolué et maintenant"

# On encode la phrase : le tokenizer la convertit en une séquence d'IDs numériques
# que le modèle peut comprendre. `return_tensors='pt'` formate la sortie pour PyTorch.
input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

print(f"\nPhrase de départ (prompt) : '{prompt_text}'")

# ------------------------------------------------------------------------------
#  ÉTAPE 5 : Génération du texte avec le modèle
# ------------------------------------------------------------------------------
# On utilise la fonction `generate()` du modèle pour créer la suite du texte.
print("\nGénération du texte en cours...")

# Paramètres de la génération :
# - input_ids : notre phrase de départ encodée.
# - max_length : la longueur totale maximale du texte généré (prompt inclus).
# - num_return_sequences : le nombre de propositions de texte à générer.
# - no_repeat_ngram_size=2 : empêche le modèle de répéter les mêmes groupes de 2 mots,
#   ce qui améliore la qualité et la cohérence du texte.
# - temperature : ajuste la créativité. Une valeur plus basse (ex: 0.7) rend le texte
#   plus prévisible, une valeur plus haute (ex: 1.5) le rend plus créatif/aléatoire.
output_sequences = model.generate(
    input_ids,
    max_length=80,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.9
)

# ------------------------------------------------------------------------------
#  ÉTAPE 6 : Décodage et affichage du résultat
# ------------------------------------------------------------------------------
# Le modèle retourne une séquence d'IDs. Il faut la décoder pour la retransformer en texte lisible.
# On prend la première (et unique, dans notre cas) séquence générée.
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print("\n--- RÉSULTAT DE LA GÉNÉRATION ---")
print(generated_text)
print("---------------------------------")
