# file_processing.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de traitement de fichiers pour OCR Juridique
Version: 1.0 - Compatible avec votre config existante
Date: 2025-09-15
"""

import os
import re
import unicodedata
from typing import Tuple

def get_file_type(file_path: str) -> str:
    """Détermine le type de fichier."""
    if not file_path:
        return "UNKNOWN"
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return "PDF"
    elif ext in ['.txt', '.text']:
        return "TXT"
    elif ext in ['.doc', '.docx']:
        return "DOC"
    else:
        return "UNKNOWN"

def read_text_file(file_path: str) -> Tuple[str, str]:
    """Lit un fichier texte avec gestion des encodages."""
    if not os.path.exists(file_path):
        return "", "Fichier introuvable"
    
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                return content, f"Lu avec encodage {encoding}"
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return "", f"Erreur lecture: {str(e)}"
    
    return "", "Impossible de décoder le fichier"

def smart_clean(text: str, pages_texts=None) -> str:
    """Nettoyage intelligent du texte OCR."""
    if not text:
        return ""
    
    # Normalisation Unicode
    text = _normalize_unicode(text)
    
    # Suppression des caractères de contrôle
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Correction des espaces multiples
    text = re.sub(r' +', ' ', text)
    
    # Correction des sauts de ligne multiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Suppression des espaces en début/fin de ligne
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    return text.strip()

def _normalize_unicode(text: str) -> str:
    """Normalise les caractères Unicode."""
    if not text:
        return ""
    
    # Normalisation NFD puis NFC
    text = unicodedata.normalize('NFD', text)
    text = unicodedata.normalize('NFC', text)
    
    return text


# anonymization.py
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
    
    # Montants en euros (optionnel)
    amount_pattern = r'\b\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})?\s*(?:euros?|€)\b'
    amounts = re.findall(amount_pattern, text, re.IGNORECASE)
    for i, amount in enumerate(set(amounts), 1):
        anonymized = anonymized.replace(amount, f"[MONTANT_{i}]")
        replacements.append(f"{amount} → [MONTANT_{i}]")
    
    report = f"Anonymisation juridique effectuée - {len(replacements)} éléments remplacés:\n"
    report += "\n".join(replacements[:10])  # Limiter l'affichage
    if len(replacements) > 10:
        report += f"\n... et {len(replacements) - 10} autres"
    
    return anonymized, report


# cache_manager.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestionnaire de cache pour OCR
Version: 1.0 - Compatible avec votre système
Date: 2025-09-15
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any

# Compatible avec votre structure existante
CACHE_DIR = "cache"

def get_pdf_hash(file_path: str) -> Optional[str]:
    """Calcule le hash MD5 d'un fichier PDF."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.md5(content).hexdigest()
    except Exception:
        return None

def load_ocr_cache(pdf_hash: str, cleaned: bool) -> Optional[Dict[str, Any]]:
    """Charge les données OCR depuis le cache."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{pdf_hash}_{'clean' if cleaned else 'raw'}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Erreur chargement cache: {e}")
    
    return None

def save_ocr_cache(pdf_hash: str, cleaned: bool, data: Dict[str, Any]) -> bool:
    """Sauvegarde les données OCR dans le cache."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{pdf_hash}_{'clean' if cleaned else 'raw'}.json")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Erreur sauvegarde cache: {e}")
        return False

def clear_ocr_cache():
    """Nettoie le cache OCR."""
    try:
        if os.path.exists(CACHE_DIR):
            for file in os.listdir(CACHE_DIR):
                if file.endswith('.json'):
                    os.remove(os.path.join(CACHE_DIR, file))
            print("Cache OCR nettoyé")
        return True
    except Exception as e:
        print(f"Erreur nettoyage cache: {e}")
        return False

def get_cache_stats() -> str:
    """Retourne les statistiques du cache."""
    try:
        if not os.path.exists(CACHE_DIR):
            return "Aucun cache"
        
        files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
        total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)
        
        return f"{len(files)} fichiers cache, {total_size / 1024:.1f} KB"
    except Exception:
        return "Erreur calcul stats cache"
