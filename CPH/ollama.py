import gradio as gr
import requests
from pdf2image import convert_from_path
import pytesseract
import re

default_prompt = (
    "Analyse les moyens de droit présentés dans ce document juridique. "
    "Récapitule les arguments principaux et identifie les fondements juridiques invoqués."
)

def ocr_preview_structured(pdf_path, nettoyer, modele, prompt):
    try:
        images = convert_from_path(pdf_path)
        full_text = ""
        for i, image in enumerate(images[:3]):  # Limité aux 3 premières pages
            text = pytesseract.image_to_string(image, lang='fra')
            full_text += f"\n--- Page {i+1} ---\n{text}\n"

        if nettoyer:
            cleaned_text = full_text
            cleaned_text = re.sub(r'\r', '', cleaned_text)
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Paragraphes
            cleaned_text = re.sub(r'[\x00-\x08\x0B-\x1F]+', '', cleaned_text)
            cleaned_text = re.sub(r' +', ' ', cleaned_text)
            cleaned_text = re.sub(r'\n +', '\n', cleaned_text)
            cleaned_text = re.sub(r' +\n', '\n', cleaned_text)
        else:
            cleaned_text = full_text

        preview = cleaned_text.strip()

        # Appel à l'API Ollama
        payload = {
            "model": modele,
            "prompt": f"{prompt}\nTexte :\n{preview}",
            "stream": False
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        result = response.json().get("response", "Aucune réponse reçue.")

        return f"{preview}\n\n--- Analyse juridique ---\n{result}"

    except Exception as e:
        return f"Erreur : {str(e)}"

def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json().get("models", [])
        return [model["name"] for model in models]
    except Exception:
        return ["llama3", "mistral"]

models_list = get_ollama_models()

gr.Interface(
    fn=ocr_preview_structured,
    inputs=[
        gr.File(label="Uploader un PDF scanné", type="filepath"),
        gr.Checkbox(label="Nettoyage intelligent avec structure", value=True),
        gr.Dropdown(label="Modèle Ollama", choices=models_list, value=models_list[0]),
        gr.Textbox(label="Prompt juridique", value=default_prompt, lines=4)
    ],
    outputs=gr.Textbox(label="Texte extrait + Analyse juridique", lines=40),
    title="OCR structuré + Analyse juridique avec Ollama",
    description="Affiche les 3 premières pages extraites avec OCR, nettoie le texte, et applique un prompt juridique via Ollama."
).launch()

