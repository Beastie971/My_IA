#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration et utilitaires pour OCR Juridique v7
"""

import os
import sys

# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

# Prompts par défaut
DEFAULT_PROMPT_NAME = "Par défaut (analyse rédigée)"
DEFAULT_PROMPT_TEXT = """
Tu es un juriste spécialisé en droit du travail français.

Analyse le document fourni, en te concentrant sur :

    Les moyens de droit invoqués (articles de loi, conventions collectives, principes généraux du droit).
    Les demandes de la partie (prétentions chiffrées ou non).

Rédige une synthèse narrative en langage juridique français, sans utiliser de listes ni de puces pour la partie analyse. Ne reformule pas les faits. Concentre-toi uniquement sur les fondements juridiques et les demandes.

Prends le point de vue de la partie : ne dis jamais « le document… ». Utilise plutôt « le demandeur », « la partie demanderesse », « Monsieur X », « Madame X », etc.

Ne cite pas les jurisprudences (Cass…), ni les décisions de cour d’appel. Mentionne uniquement les articles de loi et conventions collectives.

Structure ta réponse en trois parties distinctes :
1. Analyse juridique

    Rédige des paragraphes introduits par un titre commençant par « Sur… ».
    Chaque paragraphe correspond à un moyen ou une demande juridique identifiable.
    Analyse en priorité la section « Discussion », mais inclue aussi les moyens présents ailleurs.

2. Liste des demandes

Ajoute ensuite la mention :
« La partie demanderesse sollicite du Conseil de Prud’hommes : »
Puis liste les demandes chiffrées ou non, en respectant leur formulation exacte.
3. MOTIVATION

Ajoute enfin une section intitulée « MOTIVATION » qui contient uniquement les titres des points de droit à trancher, tous introduits par « Sur… », par exemple :

    Sur le licenciement
    Sur la perte de chance liée au plan LTIP
    Sur les dommages et intérêts
    Sur les heures supplémentaires
    Sur la remise des documents de fin de contrat
    Sur l’article 700 du CPC et les dépens

Ne répète pas les montants ni les arguments dans cette section.
"""

EXPERT_PROMPT_TEXT = """Tu es un juriste senior en droit du travail. Rédige une analyse approfondie des moyens de droit en français juridique, SANS listes ni numérotation. Mets en évidence : (i) la qualification précise des faits, (ii) l'articulation des moyens principaux et subsidiaires, (iii) le lien exact avec les références textuelles présentes DANS le document UNIQUEMENT (si une référence manque : « non précisé dans le document »), (iv) la charge de la preuve et les incidences procédurales si le texte en fait état, (v) une conclusion motivée. Aucune invention, aucune hypothèse."""

SYSTEM_PROMPT = """Tu es juriste en droit du travail. Rédige une analyse argumentée en français juridique, sans listes à puces, en citant uniquement les fondements mentionnés dans le texte."""

# =============================================================================
# UTILITAIRES DE CHEMINS
# =============================================================================

def _script_dir():
    """Retourne le répertoire du script."""
    try:
        return os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    except:
        return os.getcwd()

def _default_store_dir():
    """Crée et retourne le répertoire des prompts."""
    script_dir = _script_dir()
    prompts_dir = os.path.join(script_dir, "prompts")
    try:
        os.makedirs(prompts_dir, exist_ok=True)
        return prompts_dir
    except:
        return script_dir

# Chemins globaux
PROMPT_STORE_DIR = _default_store_dir()
PROMPT_STORE_PATH = os.path.join(PROMPT_STORE_DIR, "prompts_store.json")

# =============================================================================
# VÉRIFICATION DES DÉPENDANCES
# =============================================================================

def check_dependencies():
    """Vérifie que toutes les dépendances nécessaires sont installées."""
    missing = []
    try:
        import gradio as gr
    except ImportError:
        missing.append("gradio")
    try:
        import requests
    except ImportError:
        missing.append("requests")
    try:
        from pdf2image import convert_from_path
    except ImportError:
        missing.append("pdf2image")
    try:
        import pytesseract
    except ImportError:
        missing.append("pytesseract")
    
    if missing:
        print(f"Erreur: Dépendances manquantes: {', '.join(missing)}")
        print("Installez avec: pip install " + " ".join(missing))
        sys.exit(1)

def calculate_text_stats(text):
    """Calcule les statistiques d'un texte."""
    if not text:
        return "Texte vide"
    
    lines = text.split('\n')
    words = len(text.split())
    chars = len(text)
    non_empty_lines = len([l for l in lines if l.strip()])
    
    return f"{chars:,} caractères | {words:,} mots | {non_empty_lines} lignes non vides | {len(lines)} lignes totales"
