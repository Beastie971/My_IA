#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Ollama et RunPod pour OCR Juridique v7.2-SYSTEM-PROMPT-FIXED
Version: 7.2-SYSTEM-PROMPT-FIXED - Assure la bonne séparation système/utilisateur
Date: 2025-01-04
"""

import json
import requests
import gradio as gr

# =============================================================================
# API OLLAMA/RUNPOD AVEC SÉPARATION SYSTÈME/UTILISATEUR CLAIRE
# =============================================================================

def generate_with_ollama(model: str, system_prompt: str, user_text: str,
                         num_ctx: int, num_predict: int, temperature: float = 0.2,
                         timeout: int = 900, ollama_url: str = "http://localhost:11434") -> str:
    """Génère une réponse avec Ollama - Version avec séparation système/utilisateur et GPU optimisé."""
    # Calculer un timeout intelligent si non spécifié
    if timeout is None:
        timeout = calculate_smart_timeout(len(user_text), model)
        print(f"Timeout calculé automatiquement : {timeout}s")
    
    url = f"{ollama_url}/api/chat"
    
    # SÉPARATION CLAIRE : prompt système vs texte utilisateur
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": num_predict,
            "num_gpu": -1,  # 🔥 FORCE UTILISATION DE TOUS LES GPUs DISPONIBLES
            "num_batch": 512,  # Augmenter la taille de batch
            "num_thread": 8,  # Parallélisation CPU
            "repeat_penalty": 1.1,
            "num_ctx": num_ctx,
            "use_mmap": True,  # Optimisation mémoire
            "use_mlock": True,  # Verrouillage mémoire pour éviter le swap
            "low_vram": False  # 🔥 MODE HAUTE PERFORMANCE GPU
        },
       "stream": False
    }
    
    print(f"🤖 Ollama - Prompt système : {len(system_prompt)} caractères")
    print(f"📄 Ollama - Texte utilisateur : {len(user_text)} caractères")
    print(f"🔥 GPU forcé avec tous les GPUs disponibles (num_gpu=-1)")
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return f"❌ Erreur : Impossible de se connecter à Ollama ({ollama_url}). Vérifiez qu'Ollama est démarré (ollama serve)."
    except requests.exceptions.Timeout:
        return f"❌ Erreur : Délai dépassé ({timeout}s)."
    except Exception as e:
        return f"❌ Erreur de connexion Ollama : {e}"
    
    if response.status_code != 200:
        error_text = response.text
        if "system memory" in error_text.lower() and "available" in error_text.lower():
            return f"❌ MÉMOIRE INSUFFISANTE : Le modèle nécessite plus de RAM que disponible.\n\n" \
                   f"Solutions :\n" \
                   f"1. Utilisez un modèle plus léger (mistral:7b-instruct ou deepseek-coder:latest)\n" \
                   f"2. Fermez d'autres applications pour libérer de la RAM\n" \
                   f"3. Redémarrez Ollama : 'ollama serve'\n\n" \
                   f"Erreur complète : {error_text}"
        return f"❌ Erreur HTTP {response.status_code} : {error_text}"

    try:
        data = response.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        elif "error" in data:
            return f"❌ Erreur Ollama : {data['error']}"
        else:
            return f"❌ Réponse inattendue : {data}"
    except json.JSONDecodeError:
        return f"❌ Erreur de décodage JSON : {response.text[:500]}"
    except Exception as e:
        return f"❌ Erreur lors de la lecture de la réponse : {e}"

def generate_with_runpod(model: str, system_prompt: str, user_text: str,
                        num_ctx: int, num_predict: int, temperature: float = 0.2,
                        timeout: int = 900, endpoint: str = "", token: str = "") -> str:
    """Génère une réponse avec RunPod API - Version avec séparation système/utilisateur."""
    if not endpoint or not token:
        return "❌ Endpoint et token RunPod requis"
    
    url = f"{endpoint}/v1/chat/completions"
    
    # SÉPARATION CLAIRE : messages système vs utilisateur
    messages = [
        {"role": "system", "content": system_prompt},  # Prompt système (instructions)
        {"role": "user", "content": user_text}         # Texte utilisateur
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": num_predict,
        "temperature": temperature,
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print(f"🤖 RunPod - Prompt système : {len(system_prompt)} caractères")
    print(f"📄 RunPod - Texte utilisateur : {len(user_text)} caractères")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        
        if response.status_code != 200:
            return f"❌ Erreur RunPod HTTP {response.status_code} : {response.text}"
        
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return f"❌ Réponse RunPod invalide : {data}"
            
    except requests.exceptions.Timeout:
        return f"❌ Erreur : Délai dépassé ({timeout}s)."
    except Exception as e:
        return f"❌ Erreur RunPod : {e}"

# =============================================================================
# FONCTIONS UTILITAIRES INCHANGÉES
# =============================================================================

def get_ollama_models(ollama_url: str = "http://localhost:11434"):
    """Récupère la liste des modèles Ollama disponibles."""
    fallback_models = [
        "mistral:7b-instruct",
        "mistral:latest", 
        "deepseek-coder:latest",
        "llama3:latest",
        "llama3.1:8b-instruct-q5_K_M"
    ]
    
    try:
        print(f"🔍 Tentative de récupération des modèles Ollama depuis {ollama_url}...")
        r = requests.get(f"{ollama_url}/api/tags", timeout=10)
        
        if r.status_code != 200:
            print(f"❌ Erreur API Ollama - Status {r.status_code}: {r.text}")
            return fallback_models
        
        data = r.json()
        models = data.get("models", [])
        names = []
        
        for m in models:
            name = m.get("name")
            if name:
                names.append(name)
        
        if not names:
            print("⚠️ Aucun modèle trouvé dans la réponse, utilisation des modèles de fallback")
            return fallback_models
        
        print(f"✅ Total: {len(names)} modèle(s) récupéré(s) depuis l'API")
        return names
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Ollama non accessible sur {ollama_url} (connexion refusée)")
        print("Vérifiez qu'Ollama est démarré avec: ollama serve")
        return fallback_models
    except Exception as e:
        print(f"❌ Erreur récupération modèles: {e}")
        return fallback_models

def refresh_models(provider, ollama_url):
    """Actualise la liste des modèles selon le fournisseur."""
    if provider == "Ollama local":
        models = get_ollama_models("http://localhost:11434")
    elif provider == "Ollama distant":
        if not ollama_url.strip():
            ollama_url = "http://localhost:11434"
        models = get_ollama_models(ollama_url)
    elif provider == "RunPod.io":
        models = [
            "meta-llama/Llama-3.1-70B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/DialoGPT-large",
            "NousResearch/Nous-Hermes-2-Yi-34B",
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct"
        ]
    else:
        models = ["Erreur: Fournisseur inconnu"]
    
    info = f"Modèles actualisés pour {provider}"
    if provider == "Ollama distant":
        info += f" ({ollama_url})"
    info += f" : {len(models)} modèle(s) trouvé(s)"
    
    return gr.update(choices=models, value=models[0] if models else ""), info

def validate_ollama_url(url: str) -> tuple:
    """Valide une URL Ollama et teste la connexion."""
    if not url.strip():
        return False, "URL vide"
    
    # Ajouter http:// si manquant
    if not url.startswith(('http://', 'https://')):
        url = f"http://{url}"
    
    try:
        import requests
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            model_count = len(data.get("models", []))
            return True, f"Connexion réussie - {model_count} modèle(s) disponible(s)"
        else:
            return False, f"Erreur HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Impossible de se connecter - vérifiez que le serveur Ollama est démarré"
    except requests.exceptions.Timeout:
        return False, "Délai de connexion dépassé"
    except Exception as e:
        return False, f"Erreur : {str(e)}"

def test_connection(provider, ollama_url, runpod_endpoint, runpod_token):
    """Test la connexion selon le fournisseur sélectionné."""
    if provider == "Ollama local":
        success, message = validate_ollama_url("http://localhost:11434")
        return f"Test connexion Ollama local : {message}"
    
    elif provider == "Ollama distant":
        if not ollama_url.strip():
            return "Test connexion : URL Ollama distante requise"
        success, message = validate_ollama_url(ollama_url)
        return f"Test connexion Ollama distant : {message}"
    
    elif provider == "RunPod.io":
        if not runpod_endpoint.strip() or not runpod_token.strip():
            return "Test connexion : Endpoint et token RunPod requis"
        
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {runpod_token}",
                "Content-Type": "application/json"
            }
            # Test simple sans vraie requête pour éviter les coûts
            url = f"{runpod_endpoint}/v1/models"
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code in [200, 404]:  # 404 acceptable pour certains endpoints
                return "Test connexion RunPod : Connexion réussie"
            else:
                return f"Test connexion RunPod : Erreur HTTP {response.status_code}"
        except Exception as e:
            return f"Test connexion RunPod : Erreur - {str(e)}"
    
    return "Test connexion : Fournisseur non reconnu"

def calculate_smart_timeout(text_length: int, model: str) -> int:
    """Calcule un timeout intelligent basé sur la longueur du texte et le modèle."""
    base_timeout = 60
    text_factor = max(1, text_length // 1000)  # 1 seconde par 1000 caractères
    
    # Facteur modèle (certains modèles sont plus lents)
    model_factors = {
        "llama": 2,
        "mistral": 1.5,
        "deepseek": 1.3,
        "default": 1
    }
    
    model_factor = model_factors.get("default", 1)
    for key, factor in model_factors.items():
        if key in model.lower():
            model_factor = factor
            break
    
    timeout = int(base_timeout + (text_factor * model_factor))
    return min(timeout, 1800)  # Maximum 30 minutes
