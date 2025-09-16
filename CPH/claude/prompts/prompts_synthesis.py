#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompts pour la synthèse et fusion - Étapes 2 et 3 du mode hybride
Optimisés pour Mixtral 8x7B ou LLaMA3.1 8B (contexte large + qualité rédactionnelle)
"""

# ÉTAPE 2: FUSION ET HARMONISATION (Mixtral 8x7B)
FUSION_PROMPTS = {
    "fusion_juridique": """FUSION ET HARMONISATION JURIDIQUE

OBJECTIF: Consolider les extractions de chunks en une analyse cohérente et structurée.

MÉTHODOLOGIE:
1. Synthétiser les éléments extraits sans redondance
2. Identifier les liens logiques entre les parties
3. Structurer une analyse juridique complète
4. Harmoniser le style et la terminologie

EXTRACTIONS À FUSIONNER:
{chunk_analyses}

PROMPT UTILISATEUR ORIGINAL:
{user_prompt}

CONSIGNES FUSION:
- Éliminer les répétitions entre chunks
- Créer une progression logique
- Maintenir la précision juridique
- Assurer la cohérence terminologique
- Identifier les contradictions éventuelles

STRUCTURE ATTENDUE:
1. CONTEXTE ET CADRE JURIDIQUE
2. ANALYSE DÉTAILLÉE PAR THÈME
3. POINTS DE CONVERGENCE/DIVERGENCE
4. ÉVALUATION CRITIQUE
5. CONCLUSIONS INTERMÉDIAIRES

FORMAT: Analyse structurée, professionnelle, sans répétition.""",

    "fusion_contractuelle": """FUSION CONTRACTUELLE - HARMONISATION

OBJECTIF: Consolider l'analyse des clauses et obligations en vision d'ensemble.

EXTRACTIONS DES CHUNKS:
{chunk_analyses}

MISSION FUSION:
1. CARTOGRAPHIE CONTRACTUELLE
   - Vue d'ensemble des obligations
   - Hiérarchisation des clauses
   - Identification des liens

2. COHÉRENCE CONTRACTUELLE
   - Compatibilité des clauses
   - Contradictions à résoudre
   - Lacunes à combler

3. ANALYSE RISQUES/OPPORTUNITÉS
   - Points de vigilance
   - Leviers négociation
   - Recommandations

PROMPT ORIGINAL: {user_prompt}

STYLE: Professionnel, structuré, actionnable.""",

    "fusion_procedurale": """FUSION PROCÉDURALE - VISION CHRONOLOGIQUE

EXTRACTIONS À HARMONISER:
{chunk_analyses}

OBJECTIF: Reconstituer la chronologie procédurale et analyser la stratégie.

FUSION ATTENDUE:
1. CHRONOLOGIE PROCÉDURALE
   - Ordre chronologique des actes
   - Respect des délais
   - Enchaînements logiques

2. STRATÉGIE DES PARTIES
   - Ligne de défense
   - Évolution des positions
   - Efficacité des moyens

3. ÉTAT PROCÉDURAL
   - Situation actuelle
   - Prochaines échéances
   - Actions recommandées

PROMPT ORIGINAL: {user_prompt}"""
}

# ÉTAPE 3: SYNTHÈSE NARRATIVE PREMIUM (LLaMA3.1 8B ou Mixtral 8x7B)
NARRATIVE_PROMPTS = {
    "synthese_executive": """SYNTHÈSE EXÉCUTIVE PREMIUM

ANALYSE FUSIONNÉE:
{fused_analysis}

MISSION: Créer une synthèse exécutive de très haute qualité rédactionnelle.

STRUCTURE PREMIUM:
1. RÉSUMÉ EXÉCUTIF (150 mots max)
   - Enjeu central en langage accessible
   - Conclusion principale
   - Recommandation prioritaire

2. ANALYSE STRATÉGIQUE
   - Forces et faiblesses du dossier
   - Opportunités et menaces
   - Scénarios probables

3. PLAN D'ACTION
   - Actions immédiates (0-30 jours)
   - Actions moyen terme (1-6 mois)
   - Surveillance à long terme

STYLE EXIGÉ:
- Clarté et précision maximales
- Phrases courtes et impactantes
- Éviter le jargon juridique excessif
- Privilégier l'aspect opérationnel
- Ton professionnel et assertif

PUBLIC CIBLE: Direction, conseil d'administration""",

    "synthese_narrative": """SYNTHÈSE NARRATIVE JURIDIQUE

ANALYSE FUSIONNÉE:
{fused_analysis}

OBJECTIF: Rédiger une synthèse narrative fluide et engageante.

APPROCHE NARRATIVE:
- Raconter "l'histoire" du dossier
- Expliquer les enjeux de manière pédagogique
- Contextualiser les éléments techniques
- Créer une progression captivante

STRUCTURE NARRATIVE:
1. MISE EN CONTEXTE
   - Situation initiale
   - Acteurs principaux
   - Enjeux émergents

2. DÉVELOPPEMENT
   - Évolution de la situation
   - Points de tension
   - Arguments et contre-arguments

3. RÉSOLUTION
   - État actuel
   - Perspectives d'évolution
   - Recommandations finales

STYLE: Accessible, fluide, professionnel mais humain.""",

    "synthese_technique": """SYNTHÈSE TECHNIQUE APPROFONDIE

ANALYSE FUSIONNÉE:
{fused_analysis}

MISSION: Synthèse technique de référence pour experts juridiques.

NIVEAU: Expert juridique
PROFONDEUR: Analyse exhaustive
PRÉCISION: Maximale

STRUCTURE TECHNIQUE:
1. CADRE JURIDIQUE APPLICABLE
   - Bases légales
   - Jurisprudence de référence
   - Doctrine pertinente

2. ANALYSE JURIDIQUE POINTUE
   - Qualification juridique précise
   - Application du droit aux faits
   - Problématiques juridiques complexes

3. ÉVALUATION CRITIQUE
   - Solidité juridique
   - Risques contentieux
   - Alternatives juridiques

4. RECOMMANDATIONS EXPERTES
   - Stratégie juridique optimale
   - Précautions procédurales
   - Veille juridique

STYLE: Rigoureux, précis, référencé."""
}

# Prompts spécialisés pour domaines spécifiques
DOMAIN_SYNTHESIS = {
    "droit_travail": """SYNTHÈSE DROIT DU TRAVAIL

FUSION: {fused_analysis}

FOCUS SPÉCIALISÉ:
1. RELATION DE TRAVAIL
   - Qualification du contrat
   - Évolution de la relation
   - Rupture et conséquences

2. CONFORMITÉ SOCIALE
   - Respect du code du travail
   - Conventions collectives
   - Représentation du personnel

3. RISQUES RH
   - Contentieux prévisibles
   - Coûts potentiels
   - Actions préventives

STYLE: Opérationnel RH""",

    "droit_commercial": """SYNTHÈSE DROIT COMMERCIAL

FUSION: {fused_analysis}

ANGLE BUSINESS:
1. OPPORTUNITÉS COMMERCIALES
   - Leviers de croissance
   - Partenariats possibles
   - Marchés accessibles

2. PROTECTION COMMERCIALE
   - Sécurisation des contrats
   - Protection IP
   - Gestion des risques

3. CONFORMITÉ BUSINESS
   - Respect de la concurrence
   - Obligations sectorielles
   - Surveillance réglementaire

STYLE: Business-friendly""",

    "droit_immobilier": """SYNTHÈSE DROIT IMMOBILIER

FUSION: {fused_analysis}

VISION PATRIMONIALE:
1. ASSET MANAGEMENT
   - Valorisation du bien
   - Optimisation fiscale
   - Stratégie patrimoniale

2. GESTION OPÉRATIONNELLE
   - Exploitation du bien
   - Maintenance juridique
   - Relations locatives

3. TRANSACTIONS
   - Opportunités cession
   - Conditions acquisition
   - Due diligence

STYLE: Conseil patrimonial"""
}

def get_fusion_prompt(prompt_type="fusion_juridique", domain=None, chunk_analyses="", user_prompt=""):
    """
    Récupère le prompt de fusion (étape 2).
    
    Args:
        prompt_type: Type de fusion
        domain: Domaine spécialisé
        chunk_analyses: Analyses des chunks à fusionner
        user_prompt: Prompt utilisateur original
    """
    
    if domain and f"fusion_{domain}" in FUSION_PROMPTS:
        template = FUSION_PROMPTS[f"fusion_{domain}"]
    elif prompt_type in FUSION_PROMPTS:
        template = FUSION_PROMPTS[prompt_type]
    else:
        template = FUSION_PROMPTS["fusion_juridique"]
    
    return template.format(
        chunk_analyses=chunk_analyses,
        user_prompt=user_prompt
    )

def get_synthesis_prompt(prompt_type="synthese_executive", domain=None, fused_analysis=""):
    """
    Récupère le prompt de synthèse narrative (étape 3).
    
    Args:
        prompt_type: Type de synthèse
        domain: Domaine spécialisé
        fused_analysis: Analyse fusionnée de l'étape 2
    """
    
    if domain and domain in DOMAIN_SYNTHESIS:
        template = DOMAIN_SYNTHESIS[domain]
    elif prompt_type in NARRATIVE_PROMPTS:
        template = NARRATIVE_PROMPTS[prompt_type]
    else:
        template = NARRATIVE_PROMPTS["synthese_executive"]
    
    return template.format(fused_analysis=fused_analysis)

# Configuration des modèles pour les étapes 2 et 3
FUSION_MODEL_CONFIG = {
    "preferred_models": ["mixtral:8x7b", "llama3.1:8b", "mixtral"],
    "temperature": 0.5,  # Équilibre créativité/précision
    "max_tokens": 4000,  # Analyses développées
    "top_p": 0.95
}

SYNTHESIS_MODEL_CONFIG = {
    "preferred_models": ["llama3.1:8b", "mixtral:8x7b", "mistral:7b"],
    "temperature": 0.7,  # Créativité pour style narratif
    "max_tokens": 3000,  # Synthèses finales
    "top_p": 0.95
}
