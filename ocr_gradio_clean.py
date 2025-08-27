import gradio as gr
from pdf2image import convert_from_path
import pytesseract
import re

def ocr_and_clean(pdf_path, nettoyer):
    try:
        images = convert_from_path(pdf_path)
        full_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang='fra')
            full_text += f"\n--- Page {i+1} ---\n{text}\n"

        if nettoyer:
            # Nettoyage du texte : suppression des sauts de ligne, espaces multiples, caractères spéciaux
            cleaned_text = re.sub(r'\s+', ' ', full_text)
            cleaned_text = re.sub(r'[^\wÀ-ÿ .,;:!?()\'\"-]', '', cleaned_text)
            return cleaned_text.strip()
        else:
            return full_text.strip()
    except Exception as e:
        return f"Erreur : {str(e)}"

gr.Interface(
    fn=ocr_and_clean,
    inputs=[
        gr.File(label="Uploader un PDF scanné", type="filepath"),
        gr.Checkbox(label="Nettoyer le texte pour intégration dans Ollama", value=True)
    ],
    outputs=gr.Textbox(label="Texte extrait", lines=30),
    title="Extraction OCR + Nettoyage pour Ollama",
    description="Extrait le texte complet d’un PDF scanné avec OCR, et propose un nettoyage pour usage dans un flux Gradio/Ollama."
).launch()

