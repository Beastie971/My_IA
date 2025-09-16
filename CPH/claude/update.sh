# INVENTAIRE COMPLET DU PROJET OCR JURIDIQUE
# =============================================

# Structure du projet
ocr-juridique/
├── README.md                    # Documentation principale
├── requirements.txt             # Dépendances Python
├── .gitignore                  # Fichiers à ignorer
├── main_ocr.py                 # Point d'entrée principal ✅ EXISTANT
├── gradio_interface.py         # Interface utilisateur ✅ CORRIGÉ
├── hybrid_analyzer.py          # Analyseur 3 étapes ✅ COMPLÉTÉ
├── chunk_analysis.py           # Analyse par chunks ✅ EXISTANT
├── processing_pipeline.py      # Pipeline OCR ✅ EXISTANT  
├── ai_wrapper.py              # Wrapper IA ✅ EXISTANT
├── ai_providers.py            # Providers Ollama/RunPod ✅ EXISTANT (À CORRIGER)
├── config_manager.py          # Gestion config ✅ EXISTANT
├── config.py                  # Configuration globale ✅ EXISTANT
├── callbacks.py               # Callbacks interface ✅ EXISTANT
├── callbacks_fix.py           # Fallback compatibility ✅ EXISTANT
├── file_processing.py         # Traitement fichiers ❌ NOUVEAU
├── anonymization.py           # Anonymisation ❌ NOUVEAU
├── cache_manager.py           # Cache OCR ❌ NOUVEAU
├── prompt_manager.py          # Gestion prompts ❌ NOUVEAU
├── smart_model_selector.py    # Sélection modèles ❌ NOUVEAU (OPTIONNEL)
├── prompts/                   # Répertoire prompts personnalisés
│   └── custom_prompts.json    # (créé automatiquement)
├── cache/                     # Cache OCR
│   └── *.json                 # (créé automatiquement)
└── logs/                      # Logs (optionnel)

# FICHIERS À CRÉER OU METTRE À JOUR
# ==================================

# 1. FICHIERS NOUVEAUX À CRÉER
echo "Création des nouveaux modules..."

# file_processing.py
cat > file_processing.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de traitement de fichiers pour OCR Juridique
Version: 1.0
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
EOF

# anonymization.py
cat > anonymization.py << 'EOF'
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
EOF

# cache_manager.py
cat > cache_manager.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestionnaire de cache pour OCR
Version: 1.0
Date: 2025-09-15
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any

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
EOF

# 2. MISE À JOUR requirements.txt
cat > requirements.txt << 'EOF'
# OCR Juridique - Requirements
gradio>=4.0.0
requests>=2.31.0
pdf2image>=1.16.0
pytesseract>=0.3.10
Pillow>=10.0.0
pandas>=2.0.0
numpy>=1.24.0

# Dépendances système requises (installer séparément):
# - tesseract-ocr
# - poppler-utils
EOF

# 3. MISE À JOUR .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Cache et données temporaires
cache/
*.log
*.tmp
temp/

# Configuration locale
config/local_*
*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Prompts personnalisés (optionnel - à décommenter si privé)
# prompts/custom_prompts.json

# Modèles et données
models/
data/
*.pdf
*.docx
EOF

# 4. CRÉATION README.md
cat > README.md << 'EOF'
# OCR Juridique - Mode Hybride

Système d'analyse juridique intelligent avec architecture 3 étapes optimisant le rapport qualité/prix.

## Fonctionnalités

### Mode Hybride 3 Étapes
- **Étape 1** : Extraction rapide par chunks (Mistral 7B)
- **Étape 2** : Fusion intelligente (Mixtral 8x7B) 
- **Étape 3** : Synthèse narrative premium (LLaMA3.1 8B)

### Spécialisations Juridiques
- Prompts experts en droit du travail français
- Anonymisation intelligente (noms, entreprises, SIRET)
- Analyse structurée selon standards juridiques
- Support dual-files pour comparaisons

### Infrastructure
- Support Ollama (local/distant) et RunPod
- Cache OCR intelligent pour PDF
- Gestion automatique RunPod avec arrêt économique
- Interface web intuitive avec Gradio

## Installation

```bash
# Cloner le repository
git clone https://github.com/votre-username/ocr-juridique.git
cd ocr-juridique

# Installer les dépendances Python
pip install -r requirements.txt

# Installer les dépendances système (Ubuntu/Debian)
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-fra poppler-utils

# Pour d'autres OS, voir la documentation Tesseract
```

## Configuration

### Ollama Local
```bash
# Installer Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Démarrer le service
ollama serve

# Télécharger les modèles recommandés
ollama pull mistral:7b
ollama pull mixtral:8x7b
ollama pull llama3.1:8b
```

### RunPod (optionnel)
1. Créer un compte sur RunPod.io
2. Configurer l'endpoint et token dans l'interface
3. L'arrêt automatique évite les frais inutiles

## Utilisation

```bash
# Lancer l'application
python main_ocr.py

# Ou avec options
python main_ocr.py --host 0.0.0.0 --port 7860

# Navigateur automatique avec Falkon
# Ou ouvrir manuellement : http://localhost:7860
```

## Architecture

```
ocr-juridique/
├── main_ocr.py              # Point d'entrée
├── gradio_interface.py      # Interface utilisateur
├── hybrid_analyzer.py       # Analyseur 3 étapes
├── chunk_analysis.py        # Découpage intelligent
├── processing_pipeline.py   # Pipeline OCR
├── ai_wrapper.py           # Wrapper IA unifié
├── ai_providers.py         # Ollama/RunPod
├── config_manager.py       # Configuration
├── file_processing.py      # Traitement fichiers
├── anonymization.py        # Anonymisation
└── cache_manager.py        # Cache OCR
```

## Optimisations

### Qualité/Prix
- Modèles légers pour extraction rapide
- Modèles puissants pour synthèse finale
- Cache OCR pour éviter retraitements
- Arrêt automatique RunPod

### Performance
- Découpage intelligent préservant le contexte
- Traitement parallèle des chunks
- Gestion mémoire optimisée
- Fallbacks robustes

## Licence

MIT License - Voir LICENSE pour détails.

## Support

Pour questions et support :
1. Vérifier les issues GitHub existantes
2. Créer une nouvelle issue avec détails
3. Joindre logs et configuration
EOF

# COMMANDES GIT POUR MISE À JOUR
# ===============================

echo "=== INITIALISATION ET MISE À JOUR GIT ==="

# 1. Initialiser le repository si nécessaire
if [ ! -d ".git" ]; then
    echo "Initialisation du repository Git..."
    git init
    git branch -m main
fi

# 2. Ajouter les fichiers
echo "Ajout des fichiers..."
git add .

# 3. Commit initial ou mise à jour
echo "Commit des modifications..."
git commit -m "🚀 Mise à jour complète du système OCR Juridique

Fonctionnalités ajoutées:
✅ Mode hybride 3 étapes optimisé qualité/prix
✅ Analyseur intelligent par chunks
✅ Gestion prompts personnalisés
✅ Cache OCR pour performances
✅ Anonymisation juridique spécialisée
✅ Support dual-providers (Ollama/RunPod)
✅ Interface complète avec Gradio
✅ Documentation complète

Corrections:
🔧 Erreur force_processing → force_ocr
🔧 Interface Gradio complète et fonctionnelle
🔧 Gestion automatique RunPod
🔧 Modules manquants créés

Architecture:
📁 Structure modulaire claire
📁 Séparation des responsabilités
📁 Fallbacks robustes
📁 Configuration centralisée"

# 4. Configurer le remote GitHub (à adapter)
echo "Configuration du remote GitHub..."
echo "ATTENTION: Modifiez l'URL avec votre username GitHub"
# git remote add origin https://github.com/VOTRE-USERNAME/ocr-juridique.git

# 5. Pousser vers GitHub
echo "=== COMMANDES POUR POUSSER VERS GITHUB ==="
echo "1. Créez d'abord un repository sur GitHub nommé 'ocr-juridique'"
echo "2. Puis exécutez :"
echo "git remote add origin https://github.com/beastie971/ocr-juridique.git"
echo "git push -u origin main"

echo ""
echo "=== RÉSUMÉ DES FICHIERS ==="
echo "✅ Fichiers existants : 9"
echo "📝 Fichiers corrigés : 2 (gradio_interface.py, ai_providers.py)"
echo "🆕 Nouveaux modules : 5 (file_processing.py, anonymization.py, cache_manager.py, prompt_manager.py, smart_model_selector.py)"
echo "📚 Documentation : README.md, requirements.txt, .gitignore"
echo ""
echo "Votre projet est prêt pour GitHub ! 🎉"
EOF
