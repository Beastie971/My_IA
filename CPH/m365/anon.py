import fitz  # PyMuPDF
import re
import gradio as gr
import os

def anonymiser_document(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file.name)
        nouveau_doc = fitz.open()
        for page in doc:
            texte = page.get_text()
            texte_anonymise = anonymiser_texte(texte)
            page_nouvelle = nouveau_doc.new_page()
            page_nouvelle.insert_text((72, 72), texte_anonymise, fontsize=11)
        output_path = "document_anonymise.pdf"
        nouveau_doc.save(output_path)
        nouveau_doc.close()
    elif ext == ".txt":
        with open(file.name, "r", encoding="utf-8") as f:
            texte = f.read()
        texte_anonymise = anonymiser_texte(texte)
        output_path = "document_anonymise.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(texte_anonymise)
    else:
        return "Format de fichier non pris en charge. Utilisez PDF ou TXT."
    return output_path

def anonymiser_texte(texte):
    texte = re.sub(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", "[Nom]", texte)
    texte = re.sub(r"\b(SARL|SA|SAS|EURL|Entreprise|Société|CCI|Chambre de Commerce)[^\n\r]*", "[Entreprise]", texte)
    texte = re.sub(r"\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*€", "[Montant]", texte)
    references = re.findall(r"\bPièce\s+n°?\d+|Article\s+[A-Z]*\s*\d+[\-\d]*|\bArrêt\s+de\s+la\s+Cour\b", texte)
    if references:
        index = "\n\n[Références détectées : " + ", ".join(references) + "]\n"
        texte += index
    return texte

interface = gr.Interface(
    fn=anonymiser_document,
    inputs=gr.File(label="Téléversez un fichier PDF ou TXT à anonymiser"),
    outputs=gr.File(label="Fichier anonymisé avec index des références"),
    title="Anonymisation Générique de Documents Juridiques",
    description="Ce script anonymise les noms, entreprises, montants et indexe les références juridiques dans les fichiers PDF ou TXT."
)

interface.launch()

