#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Hybrid Analyzer - Analyse hybride multi-modèles
Version: 1.0
Date: 2025-09-16
Auteur: Assistant IA

Ce module combine plusieurs approches d'analyse pour des résultats optimaux.
"""

import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import concurrent.futures
from threading import Lock

@dataclass
class AnalysisResult:
    """Structure pour stocker les résultats d'analyse."""
    provider: str
    model: str
    content: str
    confidence: float
    processing_time: float
    tokens_used: int
    success: bool
    error_message: Optional[str] = None

class HybridAnalyzer:
    """
    Analyseur hybride qui combine plusieurs fournisseurs/modèles
    pour des analyses robustes et complètes.
    """
    
    def __init__(self, ai_provider_manager):
        """
        Initialise l'analyseur hybride.
        
        Args:
            ai_provider_manager: Instance du gestionnaire de fournisseurs IA
        """
        self.manager = ai_provider_manager
        self.results_lock = Lock()
        self.analysis_history = []
        
        # Configurations prédéfinies pour différents types d'analyse
        self.analysis_configs = {
            "rapide": {
                "primary": {"provider": "ollama_local", "model": "mistral:7b-instruct"},
                "fallback": None,
                "parallel": False
            },
            "normal": {
                "primary": {"provider": "ollama_local", "model": "llama3:latest"},
                "fallback": {"provider": "ollama_local", "model": "mistral:7b-instruct"},
                "parallel": False
            },
            "expert": {
                "primary": {"provider": "anthropic", "model": "claude-3-sonnet-20240229"},
                "fallback": {"provider": "ollama_local", "model": "llama3:latest"},
                "parallel": False
            },
            "hybride": {
                "providers": [
                    {"provider": "ollama_local", "model": "llama3:latest", "weight": 0.4},
                    {"provider": "ollama_local", "model": "mistral:7b-instruct", "weight": 0.3},
                    {"provider": "anthropic", "model": "claude-3-haiku-20240307", "weight": 0.3}
                ],
                "parallel": True,
                "synthesis_required": True
            }
        }
    
    def analyze_document(self, text: str, analysis_type: str = "normal", 
                        system_prompt: str = "", **kwargs) -> Dict:
        """
        Analyse un document avec l'approche hybride spécifiée.
        
        Args:
            text: Texte à analyser
            analysis_type: Type d'analyse (rapide, normal, expert, hybride)
            system_prompt: Prompt système
            **kwargs: Arguments supplémentaires pour les fournisseurs
            
        Returns:
            Dict contenant les résultats d'analyse
        """
        start_time = time.time()
        
        config = self.analysis_configs.get(analysis_type, self.analysis_configs["normal"])
        
        if analysis_type == "hybride":
            return self._analyze_parallel(text, config, system_prompt, **kwargs)
        else:
            return self._analyze_sequential(text, config, system_prompt, **kwargs)
    
    def _analyze_sequential(self, text: str, config: Dict, 
                           system_prompt: str, **kwargs) -> Dict:
        """Analyse séquentielle avec fallback."""
        
        # Tentative avec fournisseur principal
        primary_result = self._single_analysis(
            text, 
            config["primary"]["provider"], 
            config["primary"]["model"],
            system_prompt,
            **kwargs
        )
        
        if primary_result.success:
            return {
                "status": "success",
                "type": "sequential",
                "primary_result": primary_result,
                "final_content": primary_result.content,
                "total_time": primary_result.processing_time,
                "providers_used": [primary_result.provider]
            }
        
        # Fallback si échec du principal
        if config.get("fallback"):
            fallback_result = self._single_analysis(
                text,
                config["fallback"]["provider"],
                config["fallback"]["model"],
                system_prompt,
                **kwargs
            )
            
            return {
                "status": "fallback_used" if fallback_result.success else "failed",
                "type": "sequential",
                "primary_result": primary_result,
                "fallback_result": fallback_result,
                "final_content": fallback_result.content if fallback_result.success else "❌ Échec de tous les fournisseurs",
                "total_time": primary_result.processing_time + (fallback_result.processing_time if fallback_result else 0),
                "providers_used": [primary_result.provider, fallback_result.provider] if fallback_result else [primary_result.provider]
            }
        
        return {
            "status": "failed",
            "type": "sequential",
            "primary_result": primary_result,
            "final_content": f"❌ Échec de l'analyse : {primary_result.error_message}",
            "total_time": primary_result.processing_time,
            "providers_used": [primary_result.provider]
        }
    
    def _analyze_parallel(self, text: str, config: Dict, 
                         system_prompt: str, **kwargs) -> Dict:
        """Analyse parallèle avec plusieurs fournisseurs."""
        
        providers_config = config["providers"]
        results = []
        
        # Exécution parallèle
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers_config)) as executor:
            future_to_provider = {}
            
            for provider_config in providers_config:
                future = executor.submit(
                    self._single_analysis,
                    text,
                    provider_config["provider"],
                    provider_config["model"],
                    system_prompt,
                    **kwargs
                )
                future_to_provider[future] = provider_config
            
            # Collecte des résultats
            for future in concurrent.futures.as_completed(future_to_provider):
                provider_config = future_to_provider[future]
                try:
                    result = future.result(timeout=90)  # Timeout sécurisé
                    result.weight = provider_config["weight"]
                    results.append(result)
                except Exception as e:
                    # Créer un résultat d'échec
                    failed_result = AnalysisResult(
                        provider=provider_config["provider"],
                        model=provider_config["model"],
                        content="",
                        confidence=0.0,
                        processing_time=0.0,
                        tokens_used=0,
                        success=False,
                        error_message=str(e)
                    )
                    failed_result.weight = provider_config["weight"]
                    results.append(failed_result)
        
        # Synthèse des résultats
        if config.get("synthesis_required", True):
            synthesis_content = self._synthesize_results(results, system_prompt)
        else:
            # Sélection du meilleur résultat
            successful_results = [r for r in results if r.success]
            if successful_results:
                best_result = max(successful_results, key=lambda x: x.confidence * x.weight)
                synthesis_content = best_result.content
            else:
                synthesis_content = "❌ Aucun fournisseur n'a réussi l'analyse"
        
        total_time = max([r.processing_time for r in results], default=0)
        successful_count = sum(1 for r in results if r.success)
        
        return {
            "status": "success" if successful_count > 0 else "failed",
            "type": "parallel",
            "results": results,
            "final_content": synthesis_content,
            "total_time": total_time,
            "providers_used": [r.provider for r in results],
            "success_rate": f"{successful_count}/{len(results)}",
            "synthesis_applied": config.get("synthesis_required", True)
        }
    
    def _single_analysis(self, text: str, provider: str, model: str, 
                        system_prompt: str, **kwargs) -> AnalysisResult:
        """Effectue une analyse avec un seul fournisseur/modèle."""
        
        start_time = time.time()
        
        try:
            # Détection automatique du chunking
            use_chunking = self._should_use_chunking(text, model, provider)
            
            if use_chunking:
                print(f"🔄 Chunking automatique activé pour {provider}/{model}")
            
            result_content = self.manager.generate(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                user_text=text,
                use_chunking=use_chunking,
                **kwargs
            )
            
            processing_time = time.time() - start_time
            
            # Évaluation de la confiance (basique)
            confidence = self._evaluate_confidence(result_content, processing_time)
            
            # Estimation des tokens (approximative)
            tokens_used = len(text.split()) + len(result_content.split())
            
            success = not result_content.startswith("❌")
            
            return AnalysisResult(
                provider=provider,
                model=model,
                content=result_content,
                confidence=confidence,
                processing_time=processing_time,
                tokens_used=tokens_used,
                success=success,
                error_message=result_content if not success else None
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return AnalysisResult(
                provider=provider,
                model=model,
                content="",
                confidence=0.0,
                processing_time=processing_time,
                tokens_used=0,
                success=False,
                error_message=str(e)
            )
    
    def _should_use_chunking(self, text: str, model: str, provider: str) -> bool:
        """Détermine si le chunking est nécessaire."""
        # Utilise la fonction existante du gestionnaire
        return self.manager.estimate_cost_and_time(provider, model, len(text))["strategy"] in ["chunking_required", "chunking_recommended"]
    
    def _evaluate_confidence(self, content: str, processing_time: float) -> float:
        """Évalue la confiance dans un résultat (méthode basique)."""
        if content.startswith("❌"):
            return 0.0
        
        # Facteurs de confiance
        length_factor = min(len(content) / 500, 1.0)  # Plus long = plus confiant
        speed_factor = max(0.1, 1.0 - (processing_time / 120))  # Trop rapide ou lent = suspect
        
        # Recherche d'indicateurs de qualité
        quality_indicators = [
            "analyse", "conclusion", "recommandation", "détail", 
            "structure", "méthodologie", "évaluation"
        ]
        
        quality_factor = sum(1 for indicator in quality_indicators 
                           if indicator.lower() in content.lower()) / len(quality_indicators)
        
        confidence = (length_factor * 0.3 + speed_factor * 0.3 + quality_factor * 0.4)
        return min(confidence, 1.0)
    
    def _synthesize_results(self, results: List[AnalysisResult], 
                           original_prompt: str) -> str:
        """Synthétise les résultats de plusieurs analyses."""
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return "❌ Aucune analyse n'a réussi pour la synthèse"
        
        if len(successful_results) == 1:
            return successful_results[0].content
        
        # Création du texte de synthèse
        synthesis_text = "SYNTHÈSE MULTI-ANALYSES\n\n"
        
        for i, result in enumerate(successful_results, 1):
            synthesis_text += f"=== ANALYSE {i} ({result.provider}/{result.model}) ===\n"
            synthesis_text += f"Confiance: {result.confidence:.2f} | Poids: {getattr(result, 'weight', 1.0)}\n"
            synthesis_text += f"Contenu:\n{result.content}\n\n"
        
        # Prompt de synthèse
        synthesis_prompt = f"""
        {original_prompt}
        
        INSTRUCTION SPÉCIALE DE SYNTHÈSE:
        Vous avez reçu plusieurs analyses du même document. 
        Votre tâche est de créer une synthèse cohérente et complète qui:
        1. Intègre les éléments les plus pertinents de chaque analyse
        2. Résout les contradictions éventuelles
        3. Fournit une conclusion unifiée et structurée
        4. Indique les points de convergence et divergence entre les analyses
        """
        
        # Utilise le meilleur fournisseur disponible pour la synthèse
        best_result = max(successful_results, key=lambda x: x.confidence * getattr(x, 'weight', 1.0))
        
        try:
            synthesis = self.manager.generate(
                provider=best_result.provider,
                model=best_result.model,
                system_prompt=synthesis_prompt,
                user_text=synthesis_text,
                use_chunking=False,  # La synthèse ne devrait pas nécessiter de chunking
                temperature=0.3  # Plus créatif pour la synthèse
            )
            
            if synthesis.startswith("❌"):
                # Fallback: retourner la meilleure analyse individuelle
                return f"SYNTHÈSE AUTOMATIQUE:\n\n{best_result.content}"
            
            return synthesis
            
        except Exception as e:
            return f"SYNTHÈSE AUTOMATIQUE (erreur: {e}):\n\n{best_result.content}"
    
    def get_analysis_stats(self) -> Dict:
        """Retourne les statistiques d'utilisation."""
        with self.results_lock:
            total_analyses = len(self.analysis_history)
            if total_analyses == 0:
                return {"message": "Aucune analyse effectuée"}
            
            successful_analyses = sum(1 for a in self.analysis_history if a.get("status") == "success")
            avg_time = sum(a.get("total_time", 0) for a in self.analysis_history) / total_analyses
            
            providers_used = {}
            for analysis in self.analysis_history:
                for provider in analysis.get("providers_used", []):
                    providers_used[provider] = providers_used.get(provider, 0) + 1
            
            return {
                "total_analyses": total_analyses,
                "success_rate": f"{successful_analyses}/{total_analyses} ({successful_analyses/total_analyses*100:.1f}%)",
                "average_time": f"{avg_time:.1f}s",
                "providers_usage": providers_used,
                "most_used_provider": max(providers_used, key=providers_used.get) if providers_used else "Aucun"
            }
    
    def save_analysis(self, analysis_result: Dict):
        """Sauvegarde l'analyse dans l'historique."""
        with self.results_lock:
            analysis_result["timestamp"] = time.time()
            self.analysis_history.append(analysis_result)
            
            # Limite l'historique à 100 analyses pour éviter la surcharge mémoire
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]

# =============================================================================
# FONCTIONS UTILITAIRES POUR L'INTÉGRATION
# =============================================================================

def create_hybrid_analyzer(ai_provider_manager):
    """Factory function pour créer un analyseur hybride."""
    return HybridAnalyzer(ai_provider_manager)

def get_available_analysis_types() -> List[str]:
    """Retourne les types d'analyse disponibles."""
    return ["rapide", "normal", "expert", "hybride"]

def get_analysis_type_description(analysis_type: str) -> str:
    """Retourne la description d'un type d'analyse."""
    descriptions = {
        "rapide": "⚡ Analyse rapide avec Mistral 7B (recommandé pour tests)",
        "normal": "🎯 Analyse standard avec Llama3 + fallback Mistral",
        "expert": "🧠 Analyse experte avec Claude Sonnet + fallback local",
        "hybride": "🔀 Analyse multi-modèles en parallèle avec synthèse automatique"
    }
    return descriptions.get(analysis_type, "Type d'analyse inconnu")

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

def example_hybrid_usage():
    """Exemple d'utilisation de l'analyseur hybride."""
    
    # Supposons que vous avez déjà un AIProviderManager
    from ai_providers import AIProviderManager
    
    manager = AIProviderManager()
    analyzer = HybridAnalyzer(manager)
    
    # Exemple de document
    document = """
    Ceci est un exemple de document juridique qui nécessite une analyse approfondie.
    Le document contient plusieurs clauses importantes qui doivent être examinées
    avec attention pour identifier les risques potentiels et les opportunités.
    """
    
    # Analyse hybride
    result = analyzer.analyze_document(
        text=document,
        analysis_type="hybride",
        system_prompt="Tu es un expert juridique. Analyse ce document en détail.",
        temperature=0.2
    )
    
    print(f"Statut: {result['status']}")
    print(f"Type: {result['type']}")
    print(f"Temps total: {result['total_time']:.1f}s")
    print(f"Fournisseurs utilisés: {result['providers_used']}")
    print(f"Contenu final: {result['final_content'][:200]}...")
    
    # Sauvegarde de l'analyse
    analyzer.save_analysis(result)
    
    # Statistiques
    stats = analyzer.get_analysis_stats()
    print(f"Statistiques: {stats}")

if __name__ == "__main__":
    example_hybrid_usage()

# ========================================
# CONSTANTES POUR L'INTERFACE GRADIO
# ========================================

# Domaines juridiques supportés
SUPPORTED_DOMAINS = {
    "Aucun": "Aucun domaine spécifique",
    "Droit du travail": "Spécialisation droit du travail",
    "Droit commercial": "Spécialisation droit commercial", 
    "Droit civil": "Spécialisation droit civil",
    "Droit pénal": "Spécialisation droit pénal",
    "Droit administratif": "Spécialisation droit administratif"
}

# Types de synthèse disponibles
SYNTHESIS_TYPES = {
    "synthese_executive": "Synthèse exécutive",
    "synthese_narrative": "Synthèse narrative complète",
    "synthese_technique": "Synthèse technique détaillée",
    "synthese_comparative": "Synthèse comparative",
    "synthese_procedurale": "Synthèse procédurale"
}

print("✅ Constantes SUPPORTED_DOMAINS et SYNTHESIS_TYPES chargées")
