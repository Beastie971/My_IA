import gradio as gr
from pdf2image import convert_from_path
import pytesseract
import re

def ocr_preview_structured(pdf_path, nettoyer):
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
            return cleaned_text.strip()
        else:
            return full_text.strip()
    except Exception as e:
        return f"Erreur : {str(e)}"

gr.Interface(
    fn=ocr_preview_structured,
    inputs=[
        gr.File(label="Uploader un PDF scanné", type="filepath"),
        gr.Checkbox(label="Nettoyage intelligent avec structure", value=True)
    ],
    outputs=gr.Textbox(label="Aperçu du texte extrait (3 pages)", lines=30),
    title="Aperçu OCR structuré + Nettoyage intelligent",
    description="Affiche les 3 premières pages extraites avec OCR en conservant la structure juridique du texte."
).launch()

