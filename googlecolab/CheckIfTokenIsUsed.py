import os
from pathlib import Path
from huggingface_hub import get_token

print("--- Début du débogage de l'authentification Hugging Face ---\n")

# Ordre de priorité de la bibliothèque :
# 1. Token passé explicitement (non applicable ici, mais pour info)
# 2. Variable d'environnement HF_TOKEN
# 3. Fichier de token dans le cache

# Étape 1 : Vérifier la variable d'environnement
print("1. Vérification de la variable d'environnement 'HF_TOKEN'...")
env_token = os.getenv("HF_TOKEN")
if env_token:
    print(f"   ✅ Trouvé ! Un token est défini dans les variables d'environnement.")
    print(f"      Le token commence par: '{env_token[:4]}...'")
else:
    print("   ❌ Non trouvé.")

print("\n" + "-"*20 + "\n")

# Étape 2 : Vérifier le fichier de token dans le cache
print("2. Vérification du fichier de token mis en cache (par `huggingface-cli login`)...")
# Le chemin standard est ~/.cache/huggingface/token
token_path = Path.home() / ".cache" / "huggingface" / "token"

if token_path.is_file():
    print(f"   ✅ Le fichier de token existe à l'emplacement : {token_path}")
    try:
        with open(token_path, "r") as f:
            # .strip() pour enlever les espaces ou sauts de ligne
            cached_token = f.read().strip()
            print(f"      Le token mis en cache commence par: '{cached_token[:4]}...'")
    except Exception as e:
        print(f"   ⚠️ Erreur en lisant le fichier : {e}")
else:
    print(f"   ❌ Le fichier de token n'a pas été trouvé à l'emplacement : {token_path}")


print("\n" + "-"*20 + "\n")

# Étape 3 : Confirmation avec la méthode officielle
print("3. Résultat final via la fonction `get_token()` de la bibliothèque...")
final_token = get_token()
if final_token:
    print("   ✅ La bibliothèque utilisera bien un token.")
    print(f"      Token qui sera utilisé (tronqué) : '{final_token[:4]}...{final_token[-4:]}'")
else:
    print("   ℹ️ La bibliothèque n'utilisera aucun token et passera en mode anonyme.")

print("\n--- Fin du débogage ---")
