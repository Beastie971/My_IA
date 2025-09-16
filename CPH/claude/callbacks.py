#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fonctions callback pour l'interface Gradio - Version Refactorisée
Version: 2.1
Date: 2025-09-12
Fonctionnalités: Callbacks modulaires pour événements UI, gestion des états
"""

import gradio as gr
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports des modules existants
try:
    from config_manager import load_ollama_config, save_url_on_change
    from ai_wrapper import ai_call_wrapper, test_ai_connection, validate_ai_params
    from chunk_analysis import ChunkAnalyzer
    from processing_pipeline import process_file_to_text
except ImportError as e:
    logger.warning(f"Import warning: {e}")
    # Définir des fonctions de fallback si nécessaire


@dataclass
class AnalysisConfig:
    """Configuration pour l'analyse."""
    modele: str
    profil: str
    max_tokens_out: int
    prompt_text: str
    mode_analysis: str
    temperature: float
    top_p: float
    provider: str
    ollama_url_val: str
    runpod_endpoint: str
    runpod_token: str
    enable_chunks: bool
    chunk_size: int
    chunk_overlap: int
    synthesis_prompt: str
    enable_global_synthesis: bool

    def __post_init__(self):
        """Validation des paramètres après initialisation."""
        if self.chunk_size < 100:
            logger.warning("Chunk size très petit, ajusté à 100")
            self.chunk_size = 100
        if not (0 <= self.temperature <= 2):
            logger.warning("Température hors limites, ajustée")
            self.temperature = max(0, min(2, self.temperature))


@dataclass
class DocumentData:
    """Données des documents à analyser."""
    text1: str
    text2: str
    file_path1: str
    file_path2: str
    nettoyer: bool
    anonymiser: bool
    processing_mode: str


@dataclass
class AnalysisResult:
    """Résultat d'analyse standardisé."""
    resultat_final: str
    stats1: str
    text1: str
    preview1: str
    stats2: str
    text2: str
    preview2: str
    current_text1: str
    current_text2: str
    current_file_path1: str
    current_file_path2: str
    debug_info: str
    chunk_report: str

    def to_gradio_tuple(self) -> Tuple:
        """Convertit le résultat au format attendu par Gradio."""
        return (
            self.resultat_final, self.stats1, self.text1, self.preview1,
            self.stats2, self.text2, self.preview2, self.current_text1,
            self.current_text2, self.current_file_path1, self.current_file_path2,
            self.debug_info, self.chunk_report
        )


class DocumentProcessor:
    """Gestionnaire pour le traitement des documents."""
    
    @staticmethod
    def load_document_texts(doc_data: DocumentData) -> Tuple[str, str]:
        """Charge les textes des documents si nécessaire."""
        text1, text2 = doc_data.text1, doc_data.text2
        
        if not text1 and doc_data.file_path1:
            logger.info(f"ANALYSE: Chargement fichier 1: {doc_data.file_path1}")
            try:
                message, stats, text1, file_type, anon_report = process_file_to_text(
                    doc_data.file_path1, doc_data.nettoyer, doc_data.anonymiser, True
                )
                if "⛔" in message:
                    text1 = ""
                    logger.warning(f"Erreur traitement fichier 1: {message}")
            except Exception as e:
                logger.error(f"ANALYSE: Exception fichier 1: {e}")
                text1 = ""
        
        if not text2 and doc_data.file_path2:
            logger.info(f"ANALYSE: Chargement fichier 2: {doc_data.file_path2}")
            try:
                message, stats, text2, file_type, anon_report = process_file_to_text(
                    doc_data.file_path2, doc_data.nettoyer, doc_data.anonymiser, True
                )
                if "⛔" in message:
                    text2 = ""
                    logger.warning(f"Erreur traitement fichier 2: {message}")
            except Exception as e:
                logger.error(f"ANALYSE: Exception fichier 2: {e}")
                text2 = ""
        
        return text1, text2
    
    @staticmethod
    def prepare_full_text(text1: str, text2: str) -> str:
        """Prépare le texte complet pour l'analyse."""
        if text1 and text2:
            return f"=== DOCUMENT 1 ===\n{text1}\n\n=== DOCUMENT 2 ===\n{text2}"
        elif text1:
            return text1
        else:
            return text2


class ConnectionValidator:
    """Validateur de connexion AI."""
    
    @staticmethod
    def validate_configuration(config: AnalysisConfig) -> Optional[str]:
        """Valide la configuration d'analyse."""
        try:
            validation_errors = validate_ai_params(config.modele, config.prompt_text, config.provider)
            if validation_errors:
                return "ERREURS DE CONFIGURATION:\n" + "\n".join([f"- {error}" for error in validation_errors])
        except Exception as e:
            logger.error(f"Erreur validation configuration: {e}")
            return f"Erreur de validation: {str(e)}"
        return None
    
    @staticmethod
    def test_connection(config: AnalysisConfig) -> Optional[str]:
        """Test la connexion AI."""
        try:
            test_result, _ = test_ai_connection(
                config.provider, config.ollama_url_val, 
                config.runpod_endpoint, config.runpod_token
            )
            if "Erreur" in test_result or "❌" in test_result:
                return f"ERREUR CONNEXION: {test_result}\n\nVérifiez votre configuration."
        except Exception as e:
            logger.error(f"Erreur test connexion: {e}")
            return f"ERREUR TEST CONNEXION: {str(e)}"
        return None


class BaseAnalysisProcessor:
    """Classe de base pour les processeurs d'analyse."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def _call_ai(self, text: str, prompt: str) -> str:
        """Appel centralisé à l'IA."""
        return ai_call_wrapper(
            text=text,
            prompt=prompt,
            modele=self.config.modele,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens_out=self.config.max_tokens_out,
            provider=self.config.provider,
            ollama_url_val=self.config.ollama_url_val,
            runpod_endpoint=self.config.runpod_endpoint,
            runpod_token=self.config.runpod_token
        )
    
    def _get_provider_name(self) -> str:
        """Retourne le nom du fournisseur formaté."""
        return "RunPod" if self.config.provider == "RunPod.io" else "Ollama"


class ChunkAnalysisProcessor(BaseAnalysisProcessor):
    """Processeur pour l'analyse par chunks."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.chunk_analyzer = ChunkAnalyzer(
            chunk_size=config.chunk_size, 
            overlap=config.chunk_overlap
        )
    
    def process(self, full_text: str, current_time: str) -> Tuple[str, str, str]:
        """Traite l'analyse par chunks."""
        logger.info(f"ANALYSE: Mode chunks activé - {len(full_text)} caractères")
        
        # Découper en chunks
        chunks = self.chunk_analyzer.smart_chunk_text(full_text, preserve_structure=True)
        logger.info(f"ANALYSE: {len(chunks)} chunks créés")
        
        # Analyser chaque chunk
        analyzed_chunks = self.chunk_analyzer.analyze_chunks_with_ai(
            chunks=chunks,
            prompt=self.config.prompt_text.strip(),
            ai_function=ai_call_wrapper,
            modele=self.config.modele,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens_out=self.config.max_tokens_out,
            provider=self.config.provider,
            ollama_url_val=self.config.ollama_url_val,
            runpod_endpoint=self.config.runpod_endpoint,
            runpod_token=self.config.runpod_token
        )
        
        # Créer la synthèse si demandée
        synthesis = self._generate_synthesis(analyzed_chunks)
        
        # Construire le résultat
        analysis_report = self.chunk_analyzer.get_analysis_report(chunks, analyzed_chunks)
        resultat_final = self._build_chunk_result(chunks, analyzed_chunks, synthesis, full_text, current_time)
        debug_info = self._build_chunk_debug_info(chunks, analyzed_chunks, full_text)
        
        return resultat_final, debug_info, analysis_report
    
    def _generate_synthesis(self, analyzed_chunks: List[Dict[str, Any]]) -> str:
        """Génère la synthèse finale."""
        if self.config.synthesis_prompt and self.config.synthesis_prompt.strip():
            logger.info("ANALYSE: Génération de la synthèse finale")
            try:
                return self.chunk_analyzer.synthesize_analyses(
                    analyzed_chunks=analyzed_chunks,
                    synthesis_prompt=self.config.synthesis_prompt.strip(),
                    ai_function=ai_call_wrapper,
                    modele=self.config.modele,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens_out=self.config.max_tokens_out,
                    provider=self.config.provider,
                    ollama_url_val=self.config.ollama_url_val,
                    runpod_endpoint=self.config.runpod_endpoint,
                    runpod_token=self.config.runpod_token
                )
            except Exception as e:
                logger.error(f"Erreur génération synthèse: {e}")
                return f"ERREUR GÉNÉRATION SYNTHÈSE: {str(e)}"
        else:
            return "AUCUN PROMPT DE SYNTHÈSE FOURNI\n\nPour obtenir une synthèse consolidée, ajoutez un prompt de synthèse dans la configuration."
    
    def _build_chunk_result(self, chunks: List, analyzed_chunks: List[Dict], synthesis: str, full_text: str, current_time: str) -> str:
        """Construit le résultat final pour l'analyse par chunks."""
        entete_chunks = f"""{'=' * 95}
                                    ANALYSE PAR CHUNKS - v8.0-REFACTORISÉ
{'=' * 95}

HORODATAGE : {current_time}
MODÈLE : {self.config.modele}
FOURNISSEUR : {self._get_provider_name()}
MODE CHUNKS : ACTIVÉ
TAILLE CHUNK : {self.config.chunk_size:,} caractères
CHEVAUCHEMENT : {self.config.chunk_overlap} caractères
NOMBRE DE CHUNKS : {len(chunks)}
TEXTE TOTAL : {len(full_text):,} caractères

{'-' * 95}
                                        SYNTHÈSE JURIDIQUE STRUCTURÉE
{'-' * 95}

{synthesis}

{'-' * 95}
                                        ANALYSES DÉTAILLÉES PAR CHUNK
{'-' * 95}
"""
        
        # Ajouter les analyses de chaque chunk
        chunk_details = []
        for chunk in analyzed_chunks:
            if chunk.get('analysis_success', False):
                chunk_details.append(f"""
=== CHUNK {chunk['id']} - ANALYSE ===
Position: {chunk['start_pos']:,} à {chunk['end_pos']:,} ({chunk['size']:,} caractères)

{chunk['analysis']}
""")
            else:
                chunk_details.append(f"""
=== CHUNK {chunk['id']} - ERREUR ===
Position: {chunk['start_pos']:,} à {chunk['end_pos']:,}

{chunk['analysis']}
""")
        
        return entete_chunks + "\n".join(chunk_details)
    
    def _build_chunk_debug_info(self, chunks: List, analyzed_chunks: List[Dict], full_text: str) -> str:
        """Construit les informations de debug pour l'analyse par chunks."""
        success_count = sum(1 for c in analyzed_chunks if c.get('analysis_success', False))
        return f"""ANALYSE PAR CHUNKS EXÉCUTÉE v8.0-REFACTORISÉ
{'=' * 60}

CONFIGURATION:
- Provider: {self.config.provider}
- Modèle: {self.config.modele}
- Chunks: {len(chunks)} créés
- Taille: {self.config.chunk_size:,} caractères
- Synthèse: {'Oui' if self.config.synthesis_prompt.strip() else 'Non'}

RÉSULTATS:
- Analyses réussies: {success_count}/{len(analyzed_chunks)}
- Texte traité: {len(full_text):,} caractères"""


class GlobalSynthesisProcessor(BaseAnalysisProcessor):
    """Processeur pour la synthèse juridique globale."""
    
    def process(self, full_text: str, current_time: str) -> Tuple[str, str, str]:
        """Traite la synthèse juridique globale."""
        logger.info("ANALYSE: Mode synthèse juridique globale activé")
        
        global_synthesis_prompt = self._build_global_prompt(full_text)
        
        try:
            # Analyse globale directe
            analyse_globale = self._call_ai(full_text, global_synthesis_prompt)
            
            resultat_final = self._build_global_result(analyse_globale, full_text, current_time)
            debug_info = self._build_global_debug_info(full_text)
            chunk_report = f"Mode synthèse juridique globale - Document de {len(full_text):,} caractères analysé avec structure juridique obligatoire"
            
            return resultat_final, debug_info, chunk_report
            
        except Exception as e:
            logger.error(f"Erreur synthèse globale: {e}")
            error_msg = f"ERREUR SYNTHÈSE GLOBALE: {str(e)}"
            return error_msg, error_msg, error_msg
    
    def _build_global_prompt(self, full_text: str) -> str:
        """Construit le prompt de synthèse juridique globale."""
        return f"""SYNTHÈSE JURIDIQUE GLOBALE

Vous êtes un juriste expert. Analysez ce document juridique et produisez une synthèse structurée selon ce plan obligatoire :

1. RÉSUMÉ EXÉCUTIF
   - Nature du document et contexte
   - Parties impliquées
   - Enjeu principal en 2-3 phrases

2. SITUATION JURIDIQUE
   - Faits pertinents
   - Contexte procédural
   - Chronologie si applicable

3. ANALYSE JURIDIQUE DÉTAILLÉE
   - Moyens et arguments développés
   - Bases légales et jurisprudentielles invoquées
   - Points de droit soulevés

4. ÉLÉMENTS CLÉS D'APPRÉCIATION
   - Arguments les plus solides
   - Points faibles ou lacunes
   - Jurisprudence applicable

5. CONCLUSIONS ET PERSPECTIVES
   - Évaluation des chances de succès
   - Recommandations pratiques
   - Suites procédurales possibles

Votre prompt d'analyse personnalisé :
{self.config.prompt_text}

DOCUMENT À ANALYSER :
{full_text}"""
    
    def _build_global_result(self, analyse_globale: str, full_text: str, current_time: str) -> str:
        """Construit le résultat pour la synthèse globale."""
        return f"""{'=' * 95}
                                    SYNTHÈSE JURIDIQUE GLOBALE - v8.0-REFACTORISÉ
{'=' * 95}

HORODATAGE : {current_time}
MODÈLE : {self.config.modele}
FOURNISSEUR : {self._get_provider_name()}
MODE : SYNTHÈSE JURIDIQUE GLOBALE
TEXTE TOTAL : {len(full_text):,} caractères

{'-' * 95}
                                    SYNTHÈSE JURIDIQUE STRUCTURÉE
{'-' * 95}

{analyse_globale}
"""
    
    def _build_global_debug_info(self, full_text: str) -> str:
        """Construit les informations de debug pour la synthèse globale."""
        return f"""SYNTHÈSE JURIDIQUE GLOBALE EXÉCUTÉE v8.0-REFACTORISÉ
{'=' * 60}

MODE: Synthèse juridique globale
TAILLE DOCUMENT: {len(full_text):,} caractères
PROVIDER: {self.config.provider}
MODÈLE: {self.config.modele}

PROMPT UTILISÉ: Synthèse juridique structurée en 5 parties + votre prompt personnel"""


class ClassicAnalysisProcessor(BaseAnalysisProcessor):
    """Processeur pour l'analyse classique."""
    
    def process(self, full_text: str, current_time: str) -> Tuple[str, str, str]:
        """Traite l'analyse classique."""
        logger.info("ANALYSE: Mode analyse classique")
        
        try:
            # Analyse classique avec le prompt personnel
            analyse = self._call_ai(full_text, self.config.prompt_text.strip())
            
            resultat_final = self._build_classic_result(analyse, full_text, current_time)
            debug_info = self._build_classic_debug_info(full_text)
            chunk_report = f"Mode analyse personnalisée - Document de {len(full_text):,} caractères traité avec votre prompt"
            
            return resultat_final, debug_info, chunk_report
            
        except Exception as e:
            logger.error(f"Erreur analyse classique: {e}")
            error_msg = f"ERREUR ANALYSE CLASSIQUE: {str(e)}"
            return error_msg, error_msg, error_msg
    
    def _build_classic_result(self, analyse: str, full_text: str, current_time: str) -> str:
        """Construit le résultat pour l'analyse classique."""
        return f"""{'=' * 95}
                                        ANALYSE PERSONNALISÉE - v8.0-REFACTORISÉ
{'=' * 95}

HORODATAGE : {current_time}
MODÈLE : {self.config.modele}
FOURNISSEUR : {self._get_provider_name()}
MODE : ANALYSE AVEC VOTRE PROMPT PERSONNEL
TEXTE TOTAL : {len(full_text):,} caractères

{'-' * 95}
                                    VOTRE PROMPT PERSONNEL
{'-' * 95}

{self.config.prompt_text}

{'-' * 95}
                                        RÉSULTAT DE L'ANALYSE
{'-' * 95}

{analyse}
"""
    
    def _build_classic_debug_info(self, full_text: str) -> str:
        """Construit les informations de debug pour l'analyse classique."""
        endpoint_info = "localhost:11434"
        if self.config.provider == 'Ollama distant':
            endpoint_info = self.config.ollama_url_val
        elif self.config.provider == 'RunPod.io':
            endpoint_info = self.config.runpod_endpoint
            
        return f"""ANALYSE PERSONNALISÉE EXÉCUTÉE v8.0-REFACTORISÉ
{'=' * 60}

CONFIGURATION COMPLÈTE:
- Provider: {self.config.provider}
- URL/Endpoint: {endpoint_info}
- Modèle: {self.config.modele}
- Température: {self.config.temperature}
- Top-p: {self.config.top_p}
- Max tokens: {self.config.max_tokens_out:,}

MODE: Analyse avec votre prompt personnel
TAILLE DOCUMENT: {len(full_text):,} caractères

VOTRE PROMPT UTILISÉ:
{'-' * 40}
{self.config.prompt_text}
{'-' * 40}

RÉSULTAT: Analyse directe réalisée selon votre prompt exclusivement"""


class AnalysisOrchestrator:
    """Orchestrateur principal pour l'analyse."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.doc_processor = DocumentProcessor()
        self.connection_validator = ConnectionValidator()
    
    def analyze(self, doc_data: DocumentData) -> AnalysisResult:
        """Point d'entrée principal pour l'analyse."""
        logger.info("=" * 80)
        logger.info("ANALYSE AVEC CHUNKS v8.0-REFACTORISÉ + SYNTHESE GLOBALE")
        logger.info(f"Provider: {self.config.provider}")
        logger.info(f"Modèle: {self.config.modele}")
        logger.info(f"Chunks activés: {self.config.enable_chunks}")
        logger.info(f"Synthèse globale activée: {self.config.enable_global_synthesis}")
        logger.info("=" * 80)
        
        # Validation de la configuration
        config_error = self.connection_validator.validate_configuration(self.config)
        if config_error:
            return self._create_error_result(config_error)
        
        # Chargement des documents
        text1, text2 = self.doc_processor.load_document_texts(doc_data)
        if not text1 and not text2:
            return self._create_error_result("ERREUR: AUCUN TEXTE DISPONIBLE\nUploadez et traitez d'abord vos fichiers.")
        
        # Préparation du texte complet
        full_text = self.doc_processor.prepare_full_text(text1, text2)
        logger.info(f"ANALYSE: Texte total - {len(full_text)} caractères")
        
        # Test de connexion
        connection_error = self.connection_validator.test_connection(self.config)
        if connection_error:
            return self._create_error_result(connection_error)
        
        # Détermination du mode d'analyse et traitement
        should_use_chunks = self.config.enable_chunks and len(full_text) > self.config.chunk_size
        
        try:
            current_time = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
            
            if should_use_chunks:
                processor = ChunkAnalysisProcessor(self.config)
            elif self.config.enable_global_synthesis:
                processor = GlobalSynthesisProcessor(self.config)
            else:
                processor = ClassicAnalysisProcessor(self.config)
            
            resultat_final, debug_info, chunk_report = processor.process(full_text, current_time)
            
            # Construction du résultat final
            return self._create_success_result(
                resultat_final, debug_info, chunk_report, 
                text1, text2, doc_data.file_path1, doc_data.file_path2
            )
            
        except Exception as e:
            logger.error(f"ANALYSE: Erreur durant l'exécution: {e}")
            import traceback
            traceback.print_exc()
            
            error_msg = self._build_technical_error_message(e, full_text, should_use_chunks)
            return self._create_error_result(error_msg)
    
    def _create_error_result(self, error_msg: str) -> AnalysisResult:
        """Crée un résultat d'erreur."""
        return AnalysisResult(
            resultat_final=error_msg,
            stats1="", text1="", preview1="",
            stats2="", text2="", preview2="",
            current_text1="", current_text2="",
            current_file_path1="", current_file_path2="",
            debug_info=error_msg, chunk_report=""
        )
    
    def _create_success_result(self, resultat_final: str, debug_info: str, chunk_report: str,
                              text1: str, text2: str, file_path1: str, file_path2: str) -> AnalysisResult:
        """Crée un résultat de succès."""
        text1_length = len(text1) if text1 else 0
        text2_length = len(text2) if text2 else 0
        stats1 = f"{text1_length:,} caractères" if text1 else "Aucun texte"
        stats2 = f"{text2_length:,} caractères" if text2 else "Aucun texte"
        
        logger.info("ANALYSE: Terminée avec succès")
        
        return AnalysisResult(
            resultat_final=resultat_final,
            stats1=stats1, text1=text1 or "", preview1="",
            stats2=stats2, text2=text2 or "", preview2="",
            current_text1=text1 or "", current_text2=text2 or "",
            current_file_path1=file_path1 or "", current_file_path2=file_path2 or "",
            debug_info=debug_info, chunk_report=chunk_report
        )
    
    def _build_technical_error_message(self, e: Exception, full_text: str, should_use_chunks: bool) -> str:
        """Construit un message d'erreur technique détaillé."""
        return f"""ERREUR TECHNIQUE DURANT L'ANALYSE: {str(e)}

INFORMATIONS DE DEBUG:
- Provider: {self.config.provider}
- Modèle: {self.config.modele}
- Texte disponible: {len(full_text):,} caractères
- Mode chunks: {'Activé' if should_use_chunks else 'Désactivé'}
- Synthèse globale: {'Activée' if self.config.enable_global_synthesis else 'Désactivée'}

SOLUTIONS POSSIBLES:
1. Vérifiez que votre modèle est disponible
2. Testez la connexion
3. Essayez un modèle plus léger
4. Réduisez la taille des chunks
5. Augmentez le timeout"""


# ========================================
# FONCTIONS PRINCIPALES POUR GRADIO
# ========================================

def analyze_with_chunks_fn(text1, text2, file_path1, file_path2, modele, profil, max_tokens_out,
                          prompt_text, mode_analysis, temperature, top_p, nettoyer, anonymiser, 
                          processing_mode, provider, ollama_url_val, runpod_endpoint, runpod_token,
                          enable_chunks, chunk_size, chunk_overlap, synthesis_prompt, enable_global_synthesis):
    """Analyse avec support des chunks ET synthèse juridique globale - Version refactorisée."""
    
    try:
        # Création des objets de configuration
        config = AnalysisConfig(
            modele=modele, profil=profil, max_tokens_out=max_tokens_out,
            prompt_text=prompt_text, mode_analysis=mode_analysis, temperature=temperature,
            top_p=top_p, provider=provider, ollama_url_val=ollama_url_val,
            runpod_endpoint=runpod_endpoint, runpod_token=runpod_token,
            enable_chunks=enable_chunks, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            synthesis_prompt=synthesis_prompt, enable_global_synthesis=enable_global_synthesis
        )
        
        doc_data = DocumentData(
            text1=text1, text2=text2, file_path1=file_path1, file_path2=file_path2,
            nettoyer=nettoyer, anonymiser=anonymiser, processing_mode=processing_mode
        )
        
        # Orchestration de l'analyse
        orchestrator = AnalysisOrchestrator(config)
        result = orchestrator.analyze(doc_data)
        
        # Retour du résultat dans le format attendu par Gradio
        return result.to_gradio_tuple()
        
    except Exception as e:
        logger.error(f"Erreur dans analyze_with_chunks_fn: {e}")
        error_msg = f"ERREUR FATALE: {str(e)}"
        return (error_msg, "", "", "", "", "", "", "", "", "", "", error_msg, "")


def on_provider_change_fn(provider):
    """Gestion du changement de fournisseur."""
    try:
        ollama_visible = provider == "Ollama distant"
        runpod_visible = provider == "RunPod.io"
        
        current_ollama_url = ""
        if ollama_visible:
            try:
                current_ollama_url = load_ollama_config()
            except Exception as e:
                logger.warning(f"Erreur chargement config Ollama: {e}")
                current_ollama_url = "http://localhost:11434"
        
        status_message = ""
        if provider == "Ollama local":
            status_message = "Utilisation d'Ollama local sur http://localhost:11434"
        elif provider == "Ollama distant":
            status_message = f"URL Ollama distant: {current_ollama_url}"
        elif provider == "RunPod.io":
            status_message = "Configurez votre endpoint et token RunPod"
        
        return (
            gr.update(visible=ollama_visible, value=current_ollama_url if ollama_visible else ""),
            gr.update(visible=runpod_visible, value="" if runpod_visible else ""),
            gr.update(visible=runpod_visible, value="" if runpod_visible else ""),
            gr.update(value=status_message)
        )
        
    except Exception as e:
        logger.error(f"Erreur dans on_provider_change_fn: {e}")
        return (
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=""),
            gr.update(value=f"Erreur: {str(e)}")
        )


def on_test_connection_fn(provider, ollama_url_val, runpod_endpoint, runpod_token):
    """Test de connexion avec récupération des modèles."""
    try:
        result, models = test_ai_connection(provider, ollama_url_val, runpod_endpoint, runpod_token)
        
        if models:
            return (gr.update(choices=models, value=models[0]), gr.update(value=result))
        else:
            return (gr.update(), gr.update(value=result))
            
    except Exception as e:
        logger.error(f"Erreur dans on_test_connection_fn: {e}")
        error_msg = f"Erreur test connexion: {str(e)}"
        return (gr.update(), gr.update(value=error_msg))


def on_select_prompt_fn(name, store_dict):
    """Sélection d'un prompt utilisateur."""
    try:
        if name in store_dict:
            text = store_dict.get(name, "")
            return (
                gr.update(value=text),
                gr.update(value=f"**PROMPT SÉLECTIONNÉ :** `{name}` ({len(text)} caractères)")
            )
        else:
            return (
                gr.update(value="Prompt non trouvé."),
                gr.update(value=f"**ERREUR :** Prompt `{name}` non trouvé !")
            )
    except Exception as e:
        logger.error(f"Erreur dans on_select_prompt_fn: {e}")
        return (
            gr.update(value="Erreur lors du chargement du prompt."),
            gr.update(value=f"**ERREUR :** Exception lors du chargement: {str(e)}")
        )


def process_files_fn(file1, file2, nettoyer, anonymiser, force_processing, processing_mode):
    """Traitement des fichiers uploadés."""
    if not file1 and not file2:
        return ("Aucun fichier fourni", "", "", "", "", "", "", "", "", "", "", "", "")
    
    try:
        results = {}
        
        def process_single_file(file_path, file_key):
            if file_path:
                try:
                    message, stats, preview, file_type, anon_report = process_file_to_text(
                        file_path, nettoyer, anonymiser, force_processing
                    )
                    results[file_key] = (message, stats, preview, file_type, anon_report)
                except Exception as e:
                    logger.error(f"Erreur traitement fichier {file_key}: {e}")
                    results[file_key] = (f"Erreur: {str(e)}", "Erreur", "", "UNKNOWN", "")
            else:
                results[file_key] = ("Aucun fichier", "", "", "UNKNOWN", "")
        
        if file1:
            process_single_file(file1, 'file1')
        if file2:
            process_single_file(file2, 'file2')
        
        r1 = results.get('file1', ("", "", "", "UNKNOWN", ""))
        r2 = results.get('file2', ("", "", "", "UNKNOWN", ""))
        
        status_msg = []
        if file1:
            status_msg.append(f"Fichier 1: {r1[0]}")
        if file2:
            status_msg.append(f"Fichier 2: {r2[0]}")
        combined_status = "\n".join(status_msg) if status_msg else "Aucun fichier traité"
        
        return (
            combined_status, r1[1], r1[2], r1[4], r2[1], r2[2], r2[4],
            r1[2], r2[2], file1 if file1 else "", file2 if file2 else "", 
            "Fichiers traités avec succès", ""
        )
        
    except Exception as e:
        logger.error(f"Erreur dans process_files_fn: {e}")
        error_msg = f"Erreur traitement fichiers: {str(e)}"
        return (error_msg, "Erreur", "", "", "Erreur", "", "", "", "", "", "", error_msg, "")


def clear_all_states_fn():
    """Nettoie tous les états et cache."""
    logger.info("CACHE: Nettoyage complet des états")
    
    try:
        try:
            from cache_manager import clear_ocr_cache
            clear_ocr_cache()
            logger.info("CACHE: Cache OCR nettoyé")
        except ImportError:
            logger.warning("Module cache_manager non disponible")
        except Exception as e:
            logger.error(f"CACHE: Erreur nettoyage OCR: {e}")
        
        return (
            "",  # unified_analysis_box
            "",  # text1_stats  
            "",  # preview1_box
            "",  # anonymization1_report
            "",  # text2_stats
            "",  # preview2_box
            "",  # anonymization2_report
            "",  # current_text1
            "",  # current_text2
            "",  # current_file_path1
            "",  # current_file_path2
            "Cache nettoyé avec succès",  # debug_prompt_box
            ""   # chunk_report_box
        )
        
    except Exception as e:
        logger.error(f"Erreur dans clear_all_states_fn: {e}")
        error_msg = f"Erreur nettoyage cache: {str(e)}"
        return ("", "", "", "", "", "", "", "", "", "", "", error_msg, "")


def save_url_callback_fn(url):
    """Callback pour sauvegarder l'URL automatiquement."""
    try:
        if url and url.strip():
            message = save_url_on_change(url)
            return gr.update(value=message)
        return gr.update(value="URL vide")
    except Exception as e:
        logger.error(f"Erreur dans save_url_callback_fn: {e}")
        return gr.update(value=f"Erreur sauvegarde URL: {str(e)}")


# ========================================
# ALIAS POUR COMPATIBILITÉ ASCENDANTE
# ========================================

# Alias pour maintenir la compatibilité avec l'ancien nom
analyze_with_modes_fn = analyze_with_chunks_fn

# Export explicite des fonctions pour l'interface
__all__ = [
    'analyze_with_chunks_fn',
    'analyze_with_modes_fn',  # Alias pour compatibilité
    'on_provider_change_fn',
    'on_test_connection_fn', 
    'on_select_prompt_fn',
    'process_files_fn',
    'clear_all_states_fn',
    'save_url_callback_fn',
    # Classes exportées pour usage avancé
    'AnalysisConfig',
    'DocumentData', 
    'AnalysisResult',
    'AnalysisOrchestrator'
]

# Logging de démarrage
logger.info("Module callbacks refactorisé v8.0 chargé avec succès")
logger.info(f"Fonctions exportées: {len(__all__)} éléments")