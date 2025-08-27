import gradio as gr
from pdf2image import convert_from_path
import pytesseract

def ocr_from_pdf(pdf_file_path):
    try:
        # Convertir les pages du PDF en images
        images = convert_from_path(pdf_file_path)

        # Extraire le texte avec pytesseract
        extracted_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang='fra')
            extracted_text += f"\n--- Page {i+1} ---\n{text}\n"

        return extracted_text[:2000]  # Aperçu limité à 2000 caractères
    except Exception as e:
        return f"Erreur : {str(e)}"

gr.Interface(
    fn=ocr_from_pdf,
    inputs=gr.File(label="Uploader un PDF scanné", type="filepath"),
    outputs=gr.Textbox(label="Texte extrait (aperçu)", lines=30),
    title="Test OCR sur PDF scanné",
    description="Utilise pdf2image et pytesseract pour extraire le texte."
).launch()

