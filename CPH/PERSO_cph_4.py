import gradio as gr
import requests

def fetch_models(api_url):
    try:
        r = requests.get(f"{api_url}/api/tags", timeout=10)
        if r.status_code != 200:
            return [f"Erreur HTTP {r.status_code}"]
        data = r.json()
        return [m['name'] for m in data.get('models', []) if 'name' in m]
    except Exception as e:
        return [f"Erreur : {str(e)}"]

with gr.Blocks() as demo:
    gr.Markdown("## Interface IA Juridique avec Ollama")
    api_url = gr.Textbox(label="URL API Ollama", value="http://localhost:11434", interactive=True)
    model_list = gr.Dropdown(label="Modèles disponibles", choices=[], value=None)
    api_url.change(fn=fetch_models, inputs=[api_url], outputs=[model_list])
    gr.Markdown("Modèles mis à jour dynamiquement selon l'URL saisie.")

demo.launch()
