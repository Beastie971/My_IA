import gradio as gr
import requests
import json
import PyPDF2
import io
import os
import pickle
from PIL import Image
import pytesseract
import pdf2image
from typing import List, Optional, Dict

class PromptManager:
    def __init__(self, prompts_file: str = "saved_prompts.pkl"):
        self.prompts_file = prompts_file
        self.prompts = self.load_prompts()
    
    def load_prompts(self) -> Dict[str, str]:
        """Load saved prompts from file"""
        try:
            if os.path.exists(self.prompts_file):
                with open(self.prompts_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading prompts: {e}")
        
        # Default prompts
        return {
            "Analyse Juridique (Défaut)": DEFAULT_PROMPT,
            "Analyse Contractuelle": "Tu es un juriste spécialisé en droit des contrats. Analyse le contrat suivant en identifiant les clauses importantes, les obligations de chaque partie et les risques potentiels.",
            "Analyse de Jurisprudence": "Tu es un magistrat. Analyse la décision de justice suivante en identifiant les faits, la procédure, les moyens invoqués et la solution retenue par le juge."
        }
    
    def save_prompts(self):
        """Save prompts to file"""
        try:
            with open(self.prompts_file, 'wb') as f:
                pickle.dump(self.prompts, f)
        except Exception as e:
            print(f"Error saving prompts: {e}")
    
    def add_prompt(self, name: str, prompt: str):
        """Add or update a prompt"""
        self.prompts[name] = prompt
        self.save_prompts()
    
    def delete_prompt(self, name: str):
        """Delete a prompt"""
        if name in self.prompts and name != "Analyse Juridique (Défaut)":
            del self.prompts[name]
            self.save_prompts()
    
    def get_prompt_names(self) -> List[str]:
        """Get list of prompt names"""
        return list(self.prompts.keys())
    
    def get_prompt(self, name: str) -> str:
        """Get prompt by name"""
        return self.prompts.get(name, "")
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                return [model["name"] for model in models.get("models", [])]
            else:
                return ["Error: Could not connect to Ollama"]
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    def generate_response(self, model: str, prompt: str, document_content: str) -> str:
        """Generate response using Ollama API"""
        try:
            # Combine the prompt with document content
            full_prompt = f"{prompt}\n\nDocument à analyser:\n{document_content}"
            
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
            
            # If no text extracted, try alternative method
            if not text.strip():
                text = "ATTENTION: Le PDF semble être scanné ou protégé. Extraction de texte impossible avec PyPDF2."
            
            return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        try:
            # Try different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e2:
            return f"Error reading TXT file: {str(e2)}"

def process_document(file, prompt_text: str, selected_model: str, ollama_url: str) -> tuple:
    """Process uploaded document with Ollama"""
    if file is None:
        return "Veuillez télécharger un fichier.", ""
    
    if not prompt_text.strip():
        return "Veuillez saisir un prompt.", ""
    
    if not selected_model or selected_model.startswith("Error"):
        return "Veuillez sélectionner un modèle valide.", ""
    
    # Initialize Ollama interface
    ollama = OllamaInterface(ollama_url)
    
    # Extract text from file based on extension
    file_path = file.name
    if file_path.lower().endswith('.pdf'):
        document_content = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        document_content = extract_text_from_txt(file_path)
    else:
        return "Format de fichier non supporté. Utilisez PDF ou TXT.", ""
    
    if document_content.startswith("Error"):
        return document_content, ""
    
    # Debug: Show extracted content length and preview
    content_info = f"📄 Contenu extrait: {len(document_content)} caractères\n"
    if len(document_content) > 200:
        content_info += f"Aperçu: {document_content[:200]}...\n\n"
    else:
        content_info += f"Contenu complet:\n{document_content}\n\n"
    
    # Check if content is actually empty
    if not document_content.strip():
        return "❌ Le document semble vide après extraction. Vérifiez le format du fichier.", content_info
    
    # Generate response
    response = ollama.generate_response(selected_model, prompt_text, document_content)
    return response, content_info

def refresh_models(ollama_url: str) -> gr.Dropdown:
    """Refresh the list of available models"""
    ollama = OllamaInterface(ollama_url)
    models = ollama.get_available_models()
    return gr.Dropdown(choices=models, value=models[0] if models else None)

# Default legal analysis prompt from the document
DEFAULT_PROMPT = """Tu es un juge en droit du travail. Analyse uniquement la partie "Discussion" du document suivant. Identifie les moyens juridiques invoqués, en distinguant les moyens de fait et les moyens de droit.

Rédige les moyens de droit en langage juridique français, sous forme de paragraphes argumentés, en citant les textes applicables et la jurisprudence. Ne fais pas de liste. N'invente rien. Utilise uniquement les informations du document.

Chaque paragraphe doit contenir :
- Le contexte factuel
- Le fondement juridique
- L'articulation entre les faits et le droit

Utilise un style professionnel, rigoureux et synthétique."""

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Analyseur de Documents Juridiques - Ollama", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 📖 Analyseur de Documents Juridiques avec Ollama
        
        Cette interface permet d'analyser des documents juridiques en utilisant les modèles Ollama.
        Téléchargez un document PDF ou TXT, sélectionnez un modèle, et lancez l'analyse.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Configuration Ollama
                gr.Markdown("### ⚙️ Configuration Ollama")
                ollama_url = gr.Textbox(
                    label="URL Ollama",
                    value="http://localhost:11434",
                    placeholder="http://localhost:11434"
                )
                
                # Model selection
                model_dropdown = gr.Dropdown(
                    label="Modèle Ollama",
                    choices=[],
                    interactive=True
                )
                
                refresh_btn = gr.Button("🔄 Actualiser les modèles", variant="secondary")
                
                # File upload
                gr.Markdown("### 📁 Document")
                file_upload = gr.File(
                    label="Télécharger un document",
                    file_types=[".pdf", ".txt"],
                    file_count="single"
                )
                
                # Prompt input
                gr.Markdown("### 📝 Prompt d'analyse")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value=DEFAULT_PROMPT,
                    lines=10,
                    placeholder="Saisissez votre prompt d'analyse..."
                )
                
                # Process button
                process_btn = gr.Button("🚀 Analyser le document", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                # Results
                gr.Markdown("### 📋 Résultat de l'analyse")
                output_text = gr.Textbox(
                    label="Analyse juridique",
                    lines=15,
                    show_copy_button=True,
                    interactive=False
                )
                
                # Debug information
                gr.Markdown("### 🔍 Informations de débogage")
                debug_text = gr.Textbox(
                    label="Contenu extrait du document",
                    lines=8,
                    show_copy_button=True,
                    interactive=False
                )
        
        # Event handlers
        refresh_btn.click(
            fn=refresh_models,
            inputs=[ollama_url],
            outputs=[model_dropdown]
        )
        
        process_btn.click(
            fn=process_document,
            inputs=[file_upload, prompt_input, model_dropdown, ollama_url],
            outputs=[output_text, debug_text]
        )
        
        # Load models on startup
        interface.load(
            fn=refresh_models,
            inputs=[ollama_url],
            outputs=[model_dropdown]
        )
        
        # Add information section
        gr.Markdown("""
        ### ℹ️ Instructions d'utilisation
        
        1. **Configuration**: Vérifiez que l'URL Ollama est correcte (par défaut: http://localhost:11434)
        2. **Modèles**: Cliquez sur "Actualiser les modèles" pour charger la liste des modèles disponibles
        3. **Document**: Téléchargez un fichier PDF ou TXT contenant le document juridique à analyser
        4. **Prompt**: Modifiez le prompt si nécessaire (un prompt par défaut pour l'analyse juridique est fourni)
        5. **Analyse**: Cliquez sur "Analyser le document" pour lancer le traitement
        
        **Note**: Assurez-vous qu'Ollama est en cours d'exécution sur votre système et que des modèles sont installés.
        """)
    
    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
