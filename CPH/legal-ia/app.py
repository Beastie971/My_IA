import gradio as gr
import subprocess
import os
from docx import Document
import fitz  # PyMuPDF

# Fonction pour extraire le texte d'un fichier
def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file.name)
        text_content = "\n".join([page.get_text() for page in doc])
    elif ext == ".docx":
        doc = Document(file.name)
        text_content = "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".txt":
        text_content = file.read().decode("utf-8")
    else:
        return "Format de fichier non supporté."
    return text_content

# Fonction pour extraire la section "Discussion"
def extract_discussion_section(text):
    lines = text.splitlines()
    discussion_lines = []
    capture = False
    for line in lines:
        if "discussion" in line.lower():
            capture = True
        elif capture and line.strip() == "":
            break
        if capture:
            discussion_lines.append(line)
    return "\n".join(discussion_lines) if discussion_lines else text

# Fonction pour interroger le modèle Ollama
def analyse_juridique(file):
    raw_text = extract_text(file)
    discussion_text = extract_discussion_section(raw_text)
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3:8b-instruct-q5_K_M"],
            input=discussion_text.encode("utf-8"),
            capture_output=True,
            timeout=60
        )
        return result.stdout.decode("utf-8")
    except subprocess.TimeoutExpired:
        return "Le modèle a mis trop de temps à répondre."

# Interface Gradio
iface = gr.Interface(
    fn=analyse_juridique,
    inputs=gr.File(label="Téléversez un fichier juridique (PDF, DOCX, TXT)"),
    outputs=gr.Textbox(label="Analyse juridique par IA"),
    title="IA Juridique avec RTX 4090 et Meta-Llama-3",
    description="Ce modèle extrait la section 'Discussion' d'un document juridique et l'analyse avec Llama3-8B-Instruct (Q5_K_M) via Ollama."
)

iface.launch()

