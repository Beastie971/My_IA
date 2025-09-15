#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestionnaire de prompts personnalisés pour OCR Juridique
Version: 1.0
Date: 2025-09-15
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime

class PromptManager:
    """Gestionnaire pour sauvegarder et charger des prompts personnalisés."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self.prompts_file = os.path.join(prompts_dir, "custom_prompts.json")
        self.ensure_directory()
        
        # Prompts par défaut intégrés
        self.default_prompts = {
            "Analyse juridique hybride": """Analysez ce document juridique de manière approfondie :

1. IDENTIFICATION DU DOCUMENT
   - Nature et contexte
   - Parties impliquées
   - Enjeux principaux

2. ANALYSE JURIDIQUE
   - Fondements juridiques
   - Arguments développés
   - Points de droit

3. ÉVALUATION
   - Forces et faiblesses
   - Risques identifiés
   - Recommandations""",

            "Prompt par chunk": """Tu es juriste spécialisé en droit du travail. Voici un extrait d'un document de conclusions prud'homales. Analyse uniquement ce passage, sans inventer d'informations absentes.
1. Identifie les moyens de droit présents dans cet extrait et reformule-les en langage juridique français, sous forme narrative.
2. Si la section contient des arguments (discussion), résume-les de manière claire et structurée.
3. Ne conclus pas sur l'ensemble du dossier, limite-toi à ce passage.
4. N'inclus pas de noms propres ni de montants financiers.
Extrait :
[CHUNK]""",

            "Prompt de fusion": """Voici plusieurs analyses partielles issues d'un même document. Fusionne-les pour produire une synthèse unique, cohérente et sans répétitions, en respectant le style juridique demandé.
Consignes :
- Regroupe les moyens de droit en une présentation fluide et structurée.
- Intègre les arguments de la discussion en expliquant la logique des parties.
- Rédige en français juridique, sans puces ni numérotation, sous forme narrative.
- N'inclus pas de noms propres ni de montants financiers.
Analyses partielles :
[ANALYSES_CHUNKS]""",

            "Synthèse contractuelle": """Analysez ce contrat en détail :
- Obligations principales
- Clauses importantes
- Risques et garanties
- Recommandations d'amélioration""",

            "Analyse procédurale": """Examinez cette procédure :
- Chronologie des événements
- Moyens et arguments
- Respect des délais
- Prochaines étapes"""
        }
    
    def ensure_directory(self):
        """S'assure que le répertoire des prompts existe."""
        try:
            os.makedirs(self.prompts_dir, exist_ok=True)
        except Exception as e:
            print(f"Erreur création répertoire prompts: {e}")
    
    def load_prompts(self) -> Dict[str, str]:
        """Charge tous les prompts (défaut + personnalisés)."""
        all_prompts = self.default_prompts.copy()
        
        try:
            if os.path.exists(self.prompts_file):
                with open(self.prompts_file, 'r', encoding='utf-8') as f:
                    custom_prompts = json.load(f)
                    all_prompts.update(custom_prompts)
        except Exception as e:
            print(f"Erreur chargement prompts personnalisés: {e}")
        
        return all_prompts
    
    def save_prompt(self, name: str, content: str) -> bool:
        """Sauvegarde un prompt personnalisé."""
        try:
            # Charger les prompts existants
            custom_prompts = {}
            if os.path.exists(self.prompts_file):
                with open(self.prompts_file, 'r', encoding='utf-8') as f:
                    custom_prompts = json.load(f)
            
            # Ajouter/modifier le prompt
            custom_prompts[name] = {
                "content": content,
                "created": datetime.now().isoformat(),
                "type": "custom"
            }
            
            # Sauvegarder
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(custom_prompts, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Erreur sauvegarde prompt: {e}")
            return False
    
    def delete_prompt(self, name: str) -> bool:
        """Supprime un prompt personnalisé."""
        try:
            if not os.path.exists(self.prompts_file):
                return False
            
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                custom_prompts = json.load(f)
            
            if name in custom_prompts:
                del custom_prompts[name]
                
                with open(self.prompts_file, 'w', encoding='utf-8') as f:
                    json.dump(custom_prompts, f, ensure_ascii=False, indent=2)
                
                return True
            
            return False
        except Exception as e:
            print(f"Erreur suppression prompt: {e}")
            return False
    
    def get_prompt(self, name: str) -> Optional[str]:
        """Récupère un prompt spécifique."""
        all_prompts = self.load_prompts()
        
        if name in all_prompts:
            prompt_data = all_prompts[name]
            if isinstance(prompt_data, dict):
                return prompt_data.get("content", "")
            return prompt_data
        
        return None
    
    def list_prompts(self) -> List[str]:
        """Liste tous les noms de prompts disponibles."""
        all_prompts = self.load_prompts()
        return list(all_prompts.keys())
    
    def is_custom_prompt(self, name: str) -> bool:
        """Vérifie si un prompt est personnalisé (modifiable)."""
        return name not in self.default_prompts


# Instance globale
prompt_manager = PromptManager()


# Fonctions d'interface pour Gradio
def save_prompt_ui(name: str, content: str) -> str:
    """Interface pour sauvegarder un prompt depuis Gradio."""
    if not name.strip():
        return "❌ Nom de prompt requis"
    
    if not content.strip():
        return "❌ Contenu de prompt requis"
    
    # Éviter d'écraser les prompts par défaut
    if name in prompt_manager.default_prompts:
        return f"❌ Impossible de modifier le prompt par défaut '{name}'. Utilisez un autre nom."
    
    success = prompt_manager.save_prompt(name.strip(), content.strip())
    
    if success:
        return f"✅ Prompt '{name}' sauvegardé"
    else:
        return f"❌ Erreur lors de la sauvegarde de '{name}'"

def delete_prompt_ui(name: str) -> str:
    """Interface pour supprimer un prompt depuis Gradio."""
    if not name.strip():
        return "❌ Nom de prompt requis"
    
    if name in prompt_manager.default_prompts:
        return f"❌ Impossible de supprimer le prompt par défaut '{name}'"
    
    success = prompt_manager.delete_prompt(name.strip())
    
    if success:
        return f"✅ Prompt '{name}' supprimé"
    else:
        return f"❌ Prompt '{name}' introuvable ou erreur"

def load_prompt_ui(name: str) -> tuple:
    """Interface pour charger un prompt depuis Gradio."""
    if not name:
        return "", "❌ Sélectionnez un prompt"
    
    content = prompt_manager.get_prompt(name)
    
    if content:
        is_custom = prompt_manager.is_custom_prompt(name)
        status = f"✅ Prompt '{name}' chargé ({'personnalisé' if is_custom else 'par défaut'})"
        return content, status
    else:
        return "", f"❌ Prompt '{name}' introuvable"

def refresh_prompts_ui() -> tuple:
    """Actualise la liste des prompts."""
    prompts = prompt_manager.list_prompts()
    return prompts, f"✅ {len(prompts)} prompts disponibles"


# Interface Gradio pour la gestion des prompts
import gradio as gr

def create_prompt_management_interface():
    """Crée l'interface de gestion des prompts."""
    
    with gr.Group():
        gr.Markdown("### 📝 Gestion des prompts personnalisés")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_selector = gr.Dropdown(
                    choices=prompt_manager.list_prompts(),
                    label="Prompts disponibles",
                    value=""
                )
                
                with gr.Row():
                    load_btn = gr.Button("📂 Charger", size="sm")
                    refresh_btn = gr.Button("🔄 Actualiser", size="sm")
                    delete_btn = gr.Button("🗑️ Supprimer", size="sm", variant="stop")
                
                prompt_status = gr.Textbox(
                    label="Statut",
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=2):
                prompt_name = gr.Textbox(
                    label="Nom du prompt",
                    placeholder="Nom pour sauvegarder..."
                )
                
                prompt_content = gr.Textbox(
                    label="Contenu du prompt",
                    lines=10,
                    placeholder="Tapez votre prompt ici..."
                )
                
                save_btn = gr.Button("💾 Sauvegarder", variant="primary")
        
        # Événements
        load_btn.click(
            load_prompt_ui,
            inputs=[prompt_selector],
            outputs=[prompt_content, prompt_status]
        )
        
        save_btn.click(
            save_prompt_ui,
            inputs=[prompt_name, prompt_content],
            outputs=[prompt_status]
        )
        
        delete_btn.click(
            delete_prompt_ui,
            inputs=[prompt_selector],
            outputs=[prompt_status]
        )
        
        def refresh_and_update():
            prompts, status = refresh_prompts_ui()
            return gr.update(choices=prompts), status
        
        refresh_btn.click(
            refresh_and_update,
            outputs=[prompt_selector, prompt_status]
        )
    
    return prompt_selector, prompt_content, prompt_status


# Fonction d'intégration dans l'interface principale
def get_all_prompts_for_dropdown():
    """Retourne tous les prompts pour les dropdowns."""
    return prompt_manager.list_prompts()

def get_prompt_content(name: str) -> str:
    """Récupère le contenu d'un prompt."""
    content = prompt_manager.get_prompt(name)
    return content if content else ""
