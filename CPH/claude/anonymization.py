#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'anonymisation pour documents juridiques
Version: 1.0 - Spécialisé droit du travail
Date: 2025-09-15
"""

import re
from typing import Tuple

def anonymize_text(text: str) -> Tuple[str, str]:
    """Anonymise un texte juridique (droit du travail)."""
    if not text:
        return "", "Aucun texte à anonymiser"
    
    anonymized = text
    replacements = []
    
    # Noms propres (pattern amélioré pour le juridique)
    name_pattern = r'\b(?:Monsieur|Madame|M\.|Mme)\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?\b'
    names = re.findall(name_pattern, text)
    for i, name in enumerate(set(names), 1):
        anonymized = anonymized.replace(name, f"[PARTIE_{i}]")
        replacements.append(f"{name} → [PARTIE_{i}]")
    
    # Noms d'entreprises (patterns courants)
    company_patterns = [
        r'\b(?:Société|SA|SARL|SAS|EURL)\s+[A-Z][a-zA-Z\s&-]{3,}\b',
        r'\b[A-Z][a-zA-Z\s&-]{3,}\s+(?:SA|SARL|SAS|EURL)\b'
    ]
    
    for pattern in company_patterns:
        companies = re.findall(pattern, text)
        for i, company in enumerate(set(companies), 1):
            anonymized = anonymized.replace(company, f"[ENTREPRISE_{i}]")
            replacements.append(f"{company} → [ENTREPRISE_{i}]")
    
    # Numéros SIRET/SIREN
    siret_pattern = r'\b\d{14}\b'
    sirets = re.findall(siret_pattern, text)
    for i, siret in enumerate(set(sirets), 1):
        anonymized = anonymized.replace(siret, f"[SIRET_{i}]")
        replacements.append(f"{siret} → [SIRET_{i}]")
    
    # Adresses email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    for i, email in enumerate(set(emails), 1):
        anonymized = anonymized.replace(email, f"[EMAIL_{i}]")
        replacements.append(f"{email} → [EMAIL_{i}]")
    
    # Numéros de téléphone
    phone_pattern = r'\b(?:\+33|0)[1-9](?:[.\s-]?\d{2}){4}\b'
    phones = re.findall(phone_pattern, text)
    for i, phone in enumerate(set(phones), 1):
        anonymized = anonymized.replace(phone, f"[TEL_{i}]")
        replacements.append(f"{phone} → [TEL_{i}]")
    
    report = f"Anonymisation juridique effectuée - {len(replacements)} éléments remplacés:\n"
    report += "\n".join(replacements[:10])  # Limiter l'affichage
    if len(replacements) > 10:
        report += f"\n... et {len(replacements) - 10} autres"
    
    return anonymized, report
