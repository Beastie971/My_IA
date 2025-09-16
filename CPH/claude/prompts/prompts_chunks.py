#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompts pour l'analyse par chunks - Extraction rapide
Optimisés pour Mistral 7B (étape 1 du mode hybride)
"""

CHUNK_EXTRACTION_PROMPTS = {
    "juridique_extraction": """EXTRACTION JURIDIQUE - CHUNK {chunk_id}

OBJECTIF: Extraire rapidement les éléments juridiques essentiels de ce fragment.

CONSIGNE: Analysez ce fragment et extrayez UNIQUEMENT :

1. ÉLÉMENTS FACTUELS
   - Dates, montants, références
   - Parties mentionnées
   - Actions ou événements décrits

2. POINTS JURIDIQUES
   - Articles de loi cités
   - Jurisprudence mentionnée
   - Procédures évoquées

3. ARGUMENTS CLÉS
   - Moyens développés
   - Contestations soulevées
   - Défenses présentées

FORMAT: Réponse structurée, concise, factuelle.
PAS de commentaire ni d'interprétation.
FOCUS: Extraction pure des données.

FRAGMENT À ANALYSER:
{chunk_text}""",

    "contractuel_extraction": """EXTRACTION CONTRACTUELLE - CHUNK {chunk_id}

OBJECTIF: Identifier les clauses et obligations dans ce fragment.

EXTRAIRE :

1. CLAUSES IDENTIFIÉES
   - Type de clause
   - Contenu essentiel
   - Parties concernées

2. OBLIGATIONS
   - Qui doit faire quoi
   - Délais et conditions
   - Sanctions prévues

3. DROITS ET GARANTIES
   - Droits accordés
   - Garanties offertes
   - Limitations

FORMAT: Liste structurée, précise.
STYLE: Factuel, sans interprétation.

FRAGMENT:
{chunk_text}""",

    "procedure_extraction": """EXTRACTION PROCÉDURALE - CHUNK {chunk_id}

OBJECTIF: Identifier les éléments de procédure dans ce fragment.

EXTRAIRE:

1. ACTES PROCÉDURAUX
   - Type d'acte
   - Date et auteur
   - Destinataire

2. DÉLAIS ET ÉCHÉANCES
   - Délais mentionnés
   - Dates limites
   - Conséquences du non-respect

3. MOYENS ET DEMANDES
   - Demandes formulées
   - Moyens invoqués
   - Preuves apportées

STYLE: Extraction factuelle rapide.

FRAGMENT:
{chunk_text}""",

    "analyse_general": """ANALYSE RAPIDE - CHUNK {chunk_id}

Analysez ce fragment selon votre prompt principal mais de manière CONCISE.

VOTRE PROMPT:
{user_prompt}

CONSIGNE SPÉCIALE:
- Réponse structurée et courte
- Points essentiels uniquement
- Pas de développement long
- Focus sur les éléments clés

FRAGMENT:
{chunk_text}"""
}

# Prompts spécialisés par domaine
DOMAIN_SPECIFIC_CHUNKS = {
    "droit_travail": """EXTRACTION DROIT DU TRAVAIL - CHUNK {chunk_id}

EXTRAIRE:
1. RELATIONS DE TRAVAIL (contrat, licenciement, conditions)
2. TEMPS DE TRAVAIL (durée, repos, congés)
3. RÉMUNÉRATION (salaire, primes, avantages)
4. OBLIGATIONS (employeur/salarié)
5. CONFLITS (litiges, sanctions, procédures)

FRAGMENT: {chunk_text}""",

    "droit_commercial": """EXTRACTION DROIT COMMERCIAL - CHUNK {chunk_id}

EXTRAIRE:
1. OPÉRATIONS COMMERCIALES (vente, prestation, contrats)
2. SOCIÉTÉS (statuts, dirigeants, décisions)
3. CONCURRENCE (pratiques, clauses, restrictions)
4. FINANCIER (garanties, sûretés, paiements)
5. RESPONSABILITÉS (commerciale, civile, pénale)

FRAGMENT: {chunk_text}""",

    "droit_immobilier": """EXTRACTION DROIT IMMOBILIER - CHUNK {chunk_id}

EXTRAIRE:
1. BIENS (description, localisation, caractéristiques)
2. DROITS (propriété, usufruit, servitudes)
3. TRANSACTIONS (vente, location, bail)
4. URBANISME (permis, règles, contraintes)
5. COPROPRIÉTÉ (charges, assemblées, syndic)

FRAGMENT: {chunk_text}"""
}

def get_chunk_prompt(prompt_type="juridique_extraction", domain=None, user_prompt="", chunk_id=1, chunk_text=""):
    """
    Récupère et formate un prompt d'extraction pour chunk.
    
    Args:
        prompt_type: Type de prompt principal
        domain: Domaine spécialisé (optionnel)
        user_prompt: Prompt utilisateur (pour type général)
        chunk_id: Numéro du chunk
        chunk_text: Texte du chunk à analyser
    """
    
    # Prompt spécialisé par domaine
    if domain and domain in DOMAIN_SPECIFIC_CHUNKS:
        prompt_template = DOMAIN_SPECIFIC_CHUNKS[domain]
    # Prompt général
    elif prompt_type in CHUNK_EXTRACTION_PROMPTS:
        prompt_template = CHUNK_EXTRACTION_PROMPTS[prompt_type]
    else:
        prompt_template = CHUNK_EXTRACTION_PROMPTS["analyse_general"]
    
    # Formatage
    return prompt_template.format(
        chunk_id=chunk_id,
        chunk_text=chunk_text,
        user_prompt=user_prompt
    )

# Configuration des modèles pour l'étape 1
CHUNK_MODEL_CONFIG = {
    "preferred_models": ["mistral:7b", "llama3.1:8b", "mistral"],
    "temperature": 0.3,  # Précision pour extraction
    "max_tokens": 800,   # Réponses concises
    "top_p": 0.9
}
