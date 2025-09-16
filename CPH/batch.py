#!/home/seb/venv/Python3/bin/python

import subprocess
import argparse
import os
import pickle

def format_output_as_text(output: str) -> str:
    return output.replace('\n\n', '\n')

def load_input_file(file_path: str) -> str:
    if file_path.endswith('.pkl'):
        print(f"📦 Chargement du fichier pickle : {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, str):
                        return value
                raise ValueError("Aucun champ texte trouvé dans le fichier pickle.")
            elif isinstance(data, str):
                return data
            else:
                raise ValueError("Format de données non pris en charge dans le fichier pickle.")
    else:
        print(f"📄 Chargement du fichier texte : {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

def main():
    parser = argparse.ArgumentParser(description="Concatène prompt + fichier source et envoie à Ollama.")
    parser.add_argument("--model", required=True, help="Nom du modèle Ollama.")
    parser.add_argument("--prompt", required=True, help="Chemin vers le fichier de prompt.")
    parser.add_argument("--file", required=True, help="Chemin vers le fichier source à analyser.")
    parser.add_argument("--output", required=True, help="Chemin du fichier de sortie.")

    args = parser.parse_args()

    if not os.path.exists(args.prompt):
        raise FileNotFoundError(f"Fichier de prompt introuvable : {args.prompt}")
    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Fichier source introuvable : {args.file}")

    with open(args.prompt, "r", encoding="utf-8") as f_prompt:
        prompt_text = f_prompt.read()

    input_text = load_input_file(args.file)

    full_prompt = prompt_text + "\n\n" + input_text
    print("🚀 Envoi du prompt concaténé à Ollama...")
    command = [
        "ollama", "run", args.model
    ]

    try:
        result = subprocess.run(command, input=full_prompt, capture_output=True, text=True, check=True)
        raw_output = result.stdout
    except subprocess.CalledProcessError as e:
        print("❌ Erreur lors de l'exécution du modèle :")
        print(e.stderr)
        return

    formatted_output = format_output_as_text(raw_output)

    with open(args.output, "w", encoding="utf-8") as f_out:
        f.write(formatted_output)

    print(f"✅ Résultat enregistré dans : {args.output}")

if __name__ == "__main__":
    main()

