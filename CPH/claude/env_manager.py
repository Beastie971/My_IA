#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestionnaire de variables d'environnement pour OCR Juridique
Version: 1.0
Date: 2025-09-16
Gestion sécurisée des clés API via fichier .env
"""

import os
from typing import Optional, Dict
from pathlib import Path

class EnvManager:
    """Gestionnaire pour les variables d'environnement et clés API."""
    
    def __init__(self, env_file: str = ".env"):
        """
        Initialise le gestionnaire d'environnement.
        
        Args:
            env_file: Chemin vers le fichier .env
        """
        self.env_file = Path(env_file)
        self.env_vars = {}
        self.load_env()
    
    def load_env(self):
        """Charge les variables depuis le fichier .env."""
        if self.env_file.exists():
            try:
                with open(self.env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')  # Enlever les guillemets
                            self.env_vars[key] = value
                            os.environ[key] = value
                
                print(f"✅ Fichier .env chargé: {len(self.env_vars)} variables")
            except Exception as e:
                print(f"⚠️ Erreur lecture .env: {e}")
        else:
            print(f"📝 Fichier .env non trouvé: {self.env_file}")
            self.create_default_env()
    
    def create_default_env(self):
        """Crée un fichier .env par défaut avec exemples."""
        default_content = """# Configuration OCR Juridique
# Copiez ce fichier et renommez-le en .env
# Puis remplissez vos vraies clés API

# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================
OLLAMA_URL=http://localhost:11434
OLLAMA_REMOTE_URL=

# =============================================================================
# RUNPOD CONFIGURATION
# =============================================================================
RUNPOD_ENDPOINT=
RUNPOD_TOKEN=
RUNPOD_AUTO_STOP_TIMEOUT=15

# =============================================================================
# ANTHROPIC CONFIGURATION  
# =============================================================================
ANTHROPIC_API_KEY=

# =============================================================================
# OPENAI CONFIGURATION
# =============================================================================
OPENAI_API_KEY=

# =============================================================================
# AUTRES FOURNISSEURS
# =============================================================================
HUGGINGFACE_TOKEN=
REPLICATE_API_TOKEN=

# =============================================================================
# CONFIGURATION APPLICATION
# =============================================================================
APP_HOST=127.0.0.1
APP_PORT=7860
APP_DEBUG=false

# =============================================================================
# CONFIGURATION OCR
# =============================================================================
TESSERACT_CMD=
PDF_DPI=300
CACHE_ENABLED=true

# =============================================================================
# CONFIGURATION CHUNKS
# =============================================================================
DEFAULT_CHUNK_SIZE=3000
DEFAULT_CHUNK_OVERLAP=200
MAX_CONTEXT_LENGTH=8192

# =============================================================================
# SÉCURITÉ
# =============================================================================
ALLOW_FILE_UPLOAD=true
MAX_FILE_SIZE_MB=50
ALLOWED_EXTENSIONS=.pdf,.txt,.docx
"""
        
        try:
            with open(f"{self.env_file}.example", 'w', encoding='utf-8') as f:
                f.write(default_content)
            print(f"📝 Fichier .env.example créé: {self.env_file}.example")
            print("💡 Copiez-le en .env et ajoutez vos clés API")
        except Exception as e:
            print(f"❌ Erreur création .env.example: {e}")
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Récupère une variable d'environnement.
        
        Args:
            key: Nom de la variable
            default: Valeur par défaut
            
        Returns:
            Valeur de la variable ou défaut
        """
        # Priorité: variables système > fichier .env > défaut
        return os.environ.get(key) or self.env_vars.get(key) or default
    
    def set(self, key: str, value: str, save_to_file: bool = True):
        """
        Définit une variable d'environnement.
        
        Args:
            key: Nom de la variable
            value: Valeur
            save_to_file: Sauvegarder dans le fichier .env
        """
        self.env_vars[key] = value
        os.environ[key] = value
        
        if save_to_file:
            self.save_env()
    
    def save_env(self):
        """Sauvegarde les variables dans le fichier .env."""
        try:
            # Lire le contenu existant pour préserver les commentaires
            existing_lines = []
            if self.env_file.exists():
                with open(self.env_file, 'r', encoding='utf-8') as f:
                    existing_lines = f.readlines()
            
            # Reconstruire le fichier
            new_lines = []
            processed_keys = set()
            
            for line in existing_lines:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith('#') and '=' in line_stripped:
                    key = line_stripped.split('=', 1)[0].strip()
                    if key in self.env_vars:
                        new_lines.append(f"{key}={self.env_vars[key]}\n")
                        processed_keys.add(key)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            # Ajouter les nouvelles variables
            for key, value in self.env_vars.items():
                if key not in processed_keys:
                    new_lines.append(f"{key}={value}\n")
            
            with open(self.env_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            print(f"💾 Fichier .env sauvegardé")
        except Exception as e:
            print(f"❌ Erreur sauvegarde .env: {e}")
    
    def get_api_config(self) -> Dict[str, str]:
        """Retourne la configuration des API."""
        return {
            "ollama_url": self.get("OLLAMA_URL", "http://localhost:11434"),
            "ollama_remote_url": self.get("OLLAMA_REMOTE_URL", ""),
            "runpod_endpoint": self.get("RUNPOD_ENDPOINT", ""),
            "runpod_token": self.get("RUNPOD_TOKEN", ""),
            "anthropic_api_key": self.get("ANTHROPIC_API_KEY", ""),
            "openai_api_key": self.get("OPENAI_API_KEY", ""),
            "huggingface_token": self.get("HUGGINGFACE_TOKEN", ""),
            "replicate_api_token": self.get("REPLICATE_API_TOKEN", "")
        }
    
    def get_app_config(self) -> Dict:
        """Retourne la configuration de l'application."""
        return {
            "host": self.get("APP_HOST", "127.0.0.1"),
            "port": int(self.get("APP_PORT", "")),
            "debug": self.get("APP_DEBUG", "false").lower() == "true",
            "max_file_size_mb": int(self.get("MAX_FILE_SIZE_MB", "50")),
            "cache_enabled": self.get("CACHE_ENABLED", "true").lower() == "true"
        }
    
    def get_chunk_config(self) -> Dict[str, int]:
        """Retourne la configuration des chunks."""
        return {
            "chunk_size": int(self.get("DEFAULT_CHUNK_SIZE", "3000")),
            "chunk_overlap": int(self.get("DEFAULT_CHUNK_OVERLAP", "200")),
            "max_context": int(self.get("MAX_CONTEXT_LENGTH", "8192"))
        }
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Valide la présence des clés API."""
        config = self.get_api_config()
        
        validation = {
            "ollama_local": True,  # Toujours disponible
            "ollama_remote": bool(config["ollama_remote_url"]),
            "runpod": bool(config["runpod_endpoint"] and config["runpod_token"]),
            "anthropic": bool(config["anthropic_api_key"]),
            "openai": bool(config["openai_api_key"]),
            "huggingface": bool(config["huggingface_token"]),
            "replicate": bool(config["replicate_api_token"])
        }
        
        return validation
    
    def get_available_providers(self) -> list:
        """Retourne la liste des fournisseurs disponibles."""
        validation = self.validate_api_keys()
        
        providers = []
        if validation["ollama_local"]:
            providers.append("Ollama local")
        if validation["ollama_remote"]:
            providers.append("Ollama distant")
        if validation["runpod"]:
            providers.append("RunPod.io")
        if validation["anthropic"]:
            providers.append("Anthropic")
        if validation["openai"]:
            providers.append("OpenAI")
        if validation["huggingface"]:
            providers.append("HuggingFace")
        if validation["replicate"]:
            providers.append("Replicate")
        
        return providers
    
    def status_report(self) -> str:
        """Génère un rapport de statut des configurations."""
        api_config = self.get_api_config()
        app_config = self.get_app_config()
        chunk_config = self.get_chunk_config()
        validation = self.validate_api_keys()
        
        report = f"""
📊 RAPPORT DE CONFIGURATION

📁 Fichier .env: {'✅ Trouvé' if self.env_file.exists() else '❌ Non trouvé'}
🔑 Variables chargées: {len(self.env_vars)}

🤖 FOURNISSEURS IA:
- Ollama local: {'✅' if validation['ollama_local'] else '❌'} ({api_config['ollama_url']})
- Ollama distant: {'✅' if validation['ollama_remote'] else '❌'} ({api_config['ollama_remote_url'] or 'Non configuré'})
- RunPod: {'✅' if validation['runpod'] else '❌'} ({'Configuré' if validation['runpod'] else 'Token manquant'})
- Anthropic: {'✅' if validation['anthropic'] else '❌'} ({'Clé présente' if validation['anthropic'] else 'Clé manquante'})
- OpenAI: {'✅' if validation['openai'] else '❌'} ({'Clé présente' if validation['openai'] else 'Clé manquante'})

⚙️ APPLICATION:
- Host: {app_config['host']}
- Port: {app_config['port']}
- Debug: {app_config['debug']}
- Taille max fichier: {app_config['max_file_size_mb']}MB
- Cache: {'Activé' if app_config['cache_enabled'] else 'Désactivé'}

📄 CHUNKS:
- Taille par défaut: {chunk_config['chunk_size']} caractères
- Chevauchement: {chunk_config['chunk_overlap']} caractères
- Contexte max: {chunk_config['max_context']} tokens

✅ Fournisseurs disponibles: {len(self.get_available_providers())}
"""
        return report.strip()

# Instance globale
env_manager = EnvManager()

# Fonctions d'interface pour Gradio
def save_api_key_ui(provider: str, key_value: str) -> str:
    """Interface pour sauvegarder une clé API depuis Gradio."""
    
    key_mapping = {
        "RunPod Token": "RUNPOD_TOKEN",
        "RunPod Endpoint": "RUNPOD_ENDPOINT", 
        "Anthropic API Key": "ANTHROPIC_API_KEY",
        "OpenAI API Key": "OPENAI_API_KEY",
        "HuggingFace Token": "HUGGINGFACE_TOKEN",
        "Replicate Token": "REPLICATE_API_TOKEN",
        "Ollama Remote URL": "OLLAMA_REMOTE_URL"
    }
    
    if provider not in key_mapping:
        return f"❌ Provider '{provider}' non reconnu"
    
    if not key_value.strip():
        return f"❌ Valeur requise pour {provider}"
    
    try:
        env_key = key_mapping[provider]
        env_manager.set(env_key, key_value.strip())
        return f"✅ {provider} sauvegardé dans .env"
    except Exception as e:
        return f"❌ Erreur sauvegarde: {str(e)}"

def load_api_key_ui(provider: str) -> tuple:
    """Interface pour charger une clé API depuis Gradio."""
    
    key_mapping = {
        "RunPod Token": "RUNPOD_TOKEN",
        "RunPod Endpoint": "RUNPOD_ENDPOINT",
        "Anthropic API Key": "ANTHROPIC_API_KEY", 
        "OpenAI API Key": "OPENAI_API_KEY",
        "HuggingFace Token": "HUGGINGFACE_TOKEN",
        "Replicate Token": "REPLICATE_API_TOKEN",
        "Ollama Remote URL": "OLLAMA_REMOTE_URL"
    }
    
    if provider not in key_mapping:
        return "", f"❌ Provider '{provider}' non reconnu"
    
    env_key = key_mapping[provider]
    value = env_manager.get(env_key, "")
    
    if value:
        masked_value = f"{value[:8]}..." if len(value) > 8 else value
        return value, f"✅ {provider} chargé ({masked_value})"
    else:
        return "", f"⚠️ {provider} non configuré"

def get_status_report_ui() -> str:
    """Interface pour obtenir le rapport de statut."""
    return env_manager.status_report()

# Interface Gradio pour la gestion des clés API
import gradio as gr

def create_env_management_interface():
    """Crée l'interface de gestion des variables d'environnement."""
    
    with gr.Group():
        gr.Markdown("### 🔐 Gestion des clés API (.env)")
        
        with gr.Row():
            with gr.Column(scale=1):
                provider_selector = gr.Dropdown(
                    choices=[
                        "RunPod Token",
                        "RunPod Endpoint", 
                        "Anthropic API Key",
                        "OpenAI API Key",
                        "HuggingFace Token",
                        "Replicate Token",
                        "Ollama Remote URL"
                    ],
                    label="Configuration à modifier",
                    value=""
                )
                
                with gr.Row():
                    load_key_btn = gr.Button("📂 Charger", size="sm")
                    save_key_btn = gr.Button("💾 Sauvegarder", size="sm", variant="primary")
                    status_btn = gr.Button("📊 Statut", size="sm")
                
                key_status = gr.Textbox(
                    label="Statut",
                    interactive=False,
                    lines=3
                )
            
            with gr.Column(scale=2):
                key_value = gr.Textbox(
                    label="Valeur de la clé/configuration",
                    type="password",
                    placeholder="Collez votre clé API ici...",
                    lines=1
                )
                
                status_report = gr.Textbox(
                    label="Rapport de configuration",
                    interactive=False,
                    lines=12
                )
        
        # Événements
        load_key_btn.click(
            load_api_key_ui,
            inputs=[provider_selector],
            outputs=[key_value, key_status]
        )
        
        save_key_btn.click(
            save_api_key_ui,
            inputs=[provider_selector, key_value],
            outputs=[key_status]
        )
        
        status_btn.click(
            get_status_report_ui,
            outputs=[status_report]
        )
    
    return provider_selector, key_value, key_status, status_report

# Fonctions d'intégration
def get_api_config():
    """Retourne la configuration API pour l'application."""
    return env_manager.get_api_config()

def get_app_config():
    """Retourne la configuration de l'application."""
    return env_manager.get_app_config()

def get_chunk_config():
    """Retourne la configuration des chunks."""
    return env_manager.get_chunk_config()

def get_available_providers():
    """Retourne les fournisseurs disponibles."""
    return env_manager.get_available_providers()

if __name__ == "__main__":
    # Test du gestionnaire
    print("🔐 Test du gestionnaire d'environnement")
    print(env_manager.status_report())
