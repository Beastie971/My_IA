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
