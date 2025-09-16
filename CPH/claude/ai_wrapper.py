#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper pour les appels IA (Ollama et RunPod) - Version conservative
Version: 1.2 - Contextes conservatives pour stabilité maximale
Date: 2025-09-15
Fonctionnalités: Appels unifiés, gestion d'erreurs, contextes réalistes
"""

import requests
import json
from ai_providers import test_connection

def ai_call_wrapper(text, prompt, modele, temperature, top_p, max_tokens_out, 
                   provider, ollama_url_val, runpod_endpoint, runpod_token, num_ctx=None):
    """Wrapper unifié pour les appels IA avec gestion conservative du contexte."""
    
    # Détermination CONSERVATIVE du contexte selon le modèle si non spécifié
    if num_ctx is None:
        num_ctx = get_conservative_context_for_model(modele)
    
    try:
        use_runpod = provider == "RunPod.io"
        
        if use_runpod:
            print(f"AI_WRAPPER: Appel RunPod - modèle {modele} (contexte: {num_ctx})")
            from ai_providers import call_runpod_api
            response = call_runpod_api(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                model=modele,
                max_tokens=max_tokens_out,
                temperature=temperature,
                top_p=top_p,
                endpoint=runpod_endpoint,
                token=runpod_token
            )
            return response
        else:
            print(f"AI_WRAPPER: Appel Ollama - modèle {modele} (contexte conservative: {num_ctx})")
            
            # Correction URL Ollama
            if provider == "Ollama distant" and ollama_url_val and ollama_url_val.strip():
                ollama_url = ollama_url_val.strip()
            else:
                ollama_url = "http://localhost:11434"
            
            print(f"AI_WRAPPER: URL utilisée: {ollama_url}")
            
            payload = {
                "model": modele,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens_out,
                    "num_ctx": num_ctx,  # Contexte conservative adapté au modèle
                    "num_gpu": -1,    # Utiliser toute la GPU disponible
                    "num_batch": 512, # Augmenter la parallélisation
                    "repeat_penalty": 1.1
                },
                "stream": False
            }
            
            # Timeout calculé selon le modèle (plus longs pour gros modèles)
            timeout = calculate_conservative_timeout(modele, len(text))
            
            response = requests.post(
                f"{ollama_url}/api/chat",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                print(f"AI_WRAPPER: Réponse reçue - {len(content)} caractères")
                return content
            else:
                error_msg = f"ERREUR API Ollama: {response.status_code} - {response.text}"
                print(f"AI_WRAPPER: {error_msg}")
                return error_msg
                
    except requests.exceptions.Timeout:
        return f"ERREUR: Timeout - Le modèle {modele} prend trop de temps à répondre. Essayez des chunks plus petits ou un contexte réduit."
    except requests.exceptions.ConnectionError:
        return f"ERREUR: Impossible de se connecter à {provider}. Vérifiez votre configuration."
    except Exception as e:
        error_msg = f"ERREUR AI_WRAPPER: {str(e)}"
        print(error_msg)
        return error_msg

def get_conservative_context_for_model(modele: str) -> int:
    """Retourne une taille de contexte CONSERVATIVE pour éviter les crashes."""
    model_lower = modele.lower()
    
    # Configuration CONSERVATIVE pour vos modèles spécifiques
    if "mixtral:8x22b" in model_lower:
        return 8192   # CONSERVATIVE au lieu de 32K - évite les crashes mémoire
    elif "llama3.1:8b" in model_lower:
        return 8192   # CONSERVATIVE au lieu de 16K - plus stable
    elif "mistral:7b" in model_lower:
        return 4096   # Optimal pour Mistral 7B
    
    # Règles générales CONSERVATIVES pour autres modèles
    elif "70b" in model_lower or "405b" in model_lower:
        return 8192   # Conservative pour très gros modèles
    elif "22b" in model_lower or "mixtral" in model_lower:
        return 8192   # Conservative pour Mixtral variants
    elif "13b" in model_lower or "14b" in model_lower:
        return 6144   # Conservative pour modèles moyens
    elif "7b" in model_lower or "8b" in model_lower:
        return 4096   # Standard pour modèles moyens
    elif "3b" in model_lower or "1b" in model_lower:
        return 2048   # Petits modèles
    else:
        return 4096   # Valeur conservative par défaut

def calculate_conservative_timeout(modele: str, text_length: int) -> int:
    """Calcule un timeout CONSERVATIF selon le modèle et la longueur du texte."""
    base_timeout = 120  # 2 minutes de base (doublé pour stabilité)
    
    # Facteur modèle CONSERVATIF
    model_lower = modele.lower()
    if "mixtral:8x22b" in model_lower:
        model_factor = 6.0  # Mixtral 22B est TRÈS lent, timeout généreux
    elif "70b" in model_lower or "405b" in model_lower:
        model_factor = 4.0  # Gros modèles
    elif "mixtral" in model_lower or "22b" in model_lower:
        model_factor = 3.0  # Mixtral variants
    elif "13b" in model_lower or "14b" in model_lower:
        model_factor = 2.0  # Modèles moyens
    else:
        model_factor = 1.0  # Modèles légers
    
    # Facteur longueur de texte CONSERVATIF
    text_factor = max(1.0, text_length / 3000)  # Plus généreux
    
    # Calcul final
    timeout = int(base_timeout * model_factor * text_factor)
    
    # Limites min/max CONSERVATRICES
    return max(120, min(2400, timeout))  # Entre 2 minutes et 40 minutes

def test_ai_connection(provider, ollama_url_val, runpod_endpoint, runpod_token):
    """Teste la connexion IA et retourne les modèles disponibles."""
    print(f"TEST_AI: Test connexion {provider}")
    
    # Test de base
    result = test_connection(provider, ollama_url_val, runpod_endpoint, runpod_token)
    
    try:
        if provider == "RunPod.io":
            # Pour RunPod, retourner vos modèles configurés
            runpod_models = [
                "mistral:7b-instruct",
                "llama3.1:8b-instruct-q5_K_M", 
                "mixtral:8x22b-instruct-v0.1-q4_K_M"
            ]
            return result, runpod_models
        else:
            # Pour Ollama, récupérer les modèles réels
            url_to_use = ollama_url_val if provider == "Ollama distant" and ollama_url_val else "http://localhost:11434"
            
            response = requests.get(f"{url_to_use}/api/tags", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                combined_result = f"{result} | {len(models)} modèles disponibles"
                print(f"TEST_AI: {len(models)} modèles trouvés")
                return combined_result, models
            else:
                return f"{result} | Erreur récupération modèles", []
                
    except Exception as e:
        error_msg = f"{result} | Erreur: {str(e)}"
        return error_msg, []

def validate_ai_params(modele, prompt_text, provider):
    """Valide les paramètres avant l'appel IA."""
    errors = []
    
    if not prompt_text or not prompt_text.strip():
        errors.append("PROMPT VIDE - Sélectionnez un prompt dans l'onglet configuration")
    
    if not modele:
        errors.append("MODÈLE NON SÉLECTIONNÉ - Testez votre connexion et sélectionnez un modèle")
    
    # Vérifications spécifiques pour éviter les problèmes de contexte
    if "mixtral:8x22b" in modele.lower():
        if len(prompt_text) + len(str(len(prompt_text))) > 6000:
            errors.append("PROMPT TROP LONG pour Mixtral 22B - Réduisez la taille pour éviter les timeouts")
    
    return errors

def get_model_info(modele: str) -> dict:
    """Retourne les informations détaillées d'un modèle avec configuration conservative."""
    model_lower = modele.lower()
    
    info = {
        "name": modele,
        "context_size": get_conservative_context_for_model(modele),
        "estimated_speed": "rapide",
        "recommended_use": "général",
        "stability": "conservative"
    }
    
    # Informations spécifiques pour vos modèles avec contextes conservatives
    if "mixtral:8x22b" in model_lower:
        info.update({
            "estimated_speed": "lent",
            "recommended_use": "fusion complexe (contexte limité pour stabilité)",
            "strengths": "Très puissant mais nécessite contexte réduit",
            "optimal_step": 2,
            "warnings": "Utilisé avec contexte 8K au lieu de 32K pour éviter crashes",
            "memory_usage": "très élevé"
        })
    elif "llama3.1:8b" in model_lower:
        info.update({
            "estimated_speed": "moyen",
            "recommended_use": "synthèse narrative, rédaction",
            "strengths": "Excellent pour la qualité rédactionnelle",
            "optimal_step": 3,
            "stability": "stable avec contexte 8K"
        })
    elif "mistral:7b" in model_lower:
        info.update({
            "estimated_speed": "rapide",
            "recommended_use": "extraction rapide, traitement chunks",
            "strengths": "Rapide et efficace pour extraction",
            "optimal_step": 1,
            "stability": "très stable"
        })
    
    return info

def log_context_usage(modele: str, num_ctx: int, text_length: int, success: bool):
    """Log l'utilisation du contexte pour monitoring."""
    print(f"CONTEXT_LOG: {modele} | ctx:{num_ctx} | text:{text_length} chars | success:{success}")
    
    # Avertissements si approche des limites
    if num_ctx > 8192:
        print(f"WARNING: Contexte élevé ({num_ctx}) - risque de crash")
    if text_length > num_ctx * 3:  # Approximation 1 token = 3 chars
        print(f"WARNING: Texte possiblement trop long pour le contexte")
