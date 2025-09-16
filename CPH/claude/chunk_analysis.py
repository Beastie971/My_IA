#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'analyse par chunks pour documents juridiques longs - CLOUDFLARE SAFE
Version: 1.1-TIMEOUT-FIXED
Date: 2025-09-15
Fonctionnalités: Découpage optimisé timeout, chunks plus petits, parallélisation sécurisée
"""

import re
import json
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class CloudflareSafeChunkAnalyzer:
    """Analyseur de documents longs par chunks optimisé pour Cloudflare (< 85s)."""
    
    def __init__(self, chunk_size: int = 2000, overlap: int = 150, max_chunk_time: int = 30):
        """
        Initialise l'analyseur de chunks Cloudflare-safe.
        
        Args:
            chunk_size: Taille maximale d'un chunk (réduite pour rapidité)
            overlap: Chevauchement entre chunks
            max_chunk_time: Temps max par chunk en secondes
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chunk_time = max_chunk_time
        self.total_timeout = 80  # Limite globale Cloudflare
        
        self.juridical_markers = [
            "article", "alinéa", "paragraphe", "considérant", "attendu",
            "par ces motifs", "dispositif", "en conséquence", "statuant",
            "vu", "considérant que", "attendu que", "il s'ensuit",
            "partant", "dès lors", "en effet", "cependant", "toutefois",
            "néanmoins", "or", "mais", "donc", "ainsi", "par ailleurs"
        ]
        
        # Thread safety
        self._lock = threading.Lock()
        self._start_time = None
        self._cancelled = False
    
    def smart_chunk_text_safe(self, text: str, preserve_structure: bool = True) -> List[Dict]:
        """
        Découpe intelligemment le texte en chunks optimisés Cloudflare.
        
        Args:
            text: Texte à découper
            preserve_structure: Si True, évite de couper au milieu des sections juridiques
            
        Returns:
            Liste de dictionnaires contenant les chunks avec métadonnées
        """
        print(f"CHUNK_ANALYZER_SAFE: Découpage de {len(text)} caractères (Cloudflare-safe)")
        
        if len(text) <= self.chunk_size:
            return [{
                'id': 1,
                'text': text,
                'start_pos': 0,
                'end_pos': len(text),
                'size': len(text),
                'overlap_prev': 0,
                'overlap_next': 0,
                'contains_juridical': self._detect_juridical_content(text),
                'estimated_time': min(len(text) / 1000, self.max_chunk_time),  # Estimation
                'priority': 'high' if len(text) < 1500 else 'normal'
            }]
        
        chunks = []
        current_pos = 0
        chunk_id = 1
        
        # Calcul adaptatif de la taille selon la longueur totale
        adaptive_size = self._calculate_adaptive_size(len(text))
        
        while current_pos < len(text):
            # Calculer la fin du chunk avec taille adaptative
            end_pos = min(current_pos + adaptive_size, len(text))
            
            # Si on n'est pas à la fin du texte, essayer de trouver un point de coupe intelligent
            if end_pos < len(text) and preserve_structure:
                end_pos = self._find_smart_break_fast(text, current_pos, end_pos)
            
            # Extraire le chunk
            chunk_text = text[current_pos:end_pos]
            
            # Ajouter le chevauchement avec le chunk précédent si nécessaire
            overlap_prev = 0
            if chunk_id > 1 and current_pos > 0:
                overlap_start = max(0, current_pos - self.overlap)
                overlap_text = text[overlap_start:current_pos]
                chunk_text = overlap_text + chunk_text
                overlap_prev = len(overlap_text)
            
            # Calculer le chevauchement avec le chunk suivant (réduit)
            overlap_next = 0
            if end_pos < len(text):
                overlap_end = min(len(text), end_pos + min(self.overlap, 100))  # Limite overlap
                next_overlap = text[end_pos:overlap_end]
                chunk_text += next_overlap
                overlap_next = len(next_overlap)
            
            # Estimation du temps de traitement
            estimated_time = self._estimate_chunk_time(chunk_text)
            
            chunk_info = {
                'id': chunk_id,
                'text': chunk_text,
                'start_pos': current_pos,
                'end_pos': end_pos,
                'size': len(chunk_text),
                'overlap_prev': overlap_prev,
                'overlap_next': overlap_next,
                'contains_juridical': self._detect_juridical_content(chunk_text),
                'estimated_time': estimated_time,
                'priority': 'high' if estimated_time < 20 else 'normal',
                'cloudflare_safe': estimated_time < self.max_chunk_time
            }
            
            chunks.append(chunk_info)
            
            # Passer au chunk suivant
            current_pos = end_pos
            chunk_id += 1
            
            # Vérification sécurité : pas trop de chunks
            if len(chunks) > 20:  # Limite pour éviter surcharge
                print(f"⚠️ Limite de chunks atteinte (20), arrêt du découpage")
                break
        
        # Optimisation finale : fusionner les petits chunks adjacents
        optimized_chunks = self._optimize_small_chunks(chunks)
        
        print(f"CHUNK_ANALYZER_SAFE: {len(optimized_chunks)} chunks créés (optimisés Cloudflare)")
        return optimized_chunks
    
    def _calculate_adaptive_size(self, total_length: int) -> int:
        """Calcule une taille de chunk adaptative selon la longueur totale."""
        if total_length > 50000:  # Très long document
            return min(self.chunk_size, 1500)  # Chunks plus petits
        elif total_length > 20000:  # Document long
            return min(self.chunk_size, 2000)
        else:  # Document normal
            return self.chunk_size
    
    def _find_smart_break_fast(self, text: str, start: int, proposed_end: int) -> int:
        """
        Trouve un point de coupe intelligent RAPIDE pour préserver la structure juridique.
        Version optimisée pour rapidité.
        """
        # Zone de recherche réduite (200 caractères au lieu de 500)
        search_start = max(start + self.chunk_size - 200, start)
        search_text = text[search_start:proposed_end + 100]
        
        # Priorités pour les points de coupe (optimisé vitesse)
        break_patterns = [
            r'\n\n(?=[A-Z])',  # Double saut de ligne suivi d'une majuscule
            r'\n(?=Article|Considérant|Attendu)',  # Début d'article juridique (réduit)
            r'\.[\s\n]+(?=[A-Z])',  # Fin de phrase suivie d'une majuscule
            r'\n',  # Simple saut de ligne (fallback rapide)
        ]
        
        for pattern in break_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Prendre le dernier match dans la zone de recherche
                best_match = matches[-1]
                return search_start + best_match.end()
        
        # Si aucun point de coupe intelligent trouvé, utiliser la position proposée
        return proposed_end
    
    def _estimate_chunk_time(self, chunk_text: str) -> float:
        """Estime le temps de traitement d'un chunk."""
        base_time = len(chunk_text) / 1000  # 1000 caractères par seconde (estimation)
        
        # Facteur de complexité juridique
        juridical_content = self._detect_juridical_content(chunk_text)
        complexity_factor = 1.0
        
        if juridical_content['has_legal_terms'] > 5:
            complexity_factor = 1.3
        elif juridical_content['has_articles']:
            complexity_factor = 1.2
        
        estimated = base_time * complexity_factor
        return min(estimated, self.max_chunk_time)  # Plafonner
    
    def _optimize_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Fusionne les chunks trop petits pour optimiser l'efficacité."""
        if len(chunks) <= 1:
            return chunks
        
        optimized = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Si le chunk est petit et qu'il y a un suivant
            if (current_chunk['size'] < 800 and 
                i + 1 < len(chunks) and 
                chunks[i + 1]['size'] < 800):
                
                # Fusionner avec le chunk suivant
                next_chunk = chunks[i + 1]
                merged_text = current_chunk['text'] + "\n" + next_chunk['text']
                
                # Vérifier que la fusion reste dans les limites
                if len(merged_text) <= self.chunk_size * 1.5:
                    merged_chunk = {
                        'id': current_chunk['id'],
                        'text': merged_text,
                        'start_pos': current_chunk['start_pos'],
                        'end_pos': next_chunk['end_pos'],
                        'size': len(merged_text),
                        'overlap_prev': current_chunk['overlap_prev'],
                        'overlap_next': next_chunk['overlap_next'],
                        'contains_juridical': self._merge_juridical_content(
                            current_chunk['contains_juridical'],
                            next_chunk['contains_juridical']
                        ),
                        'estimated_time': self._estimate_chunk_time(merged_text),
                        'priority': 'normal',
                        'cloudflare_safe': True,
                        'merged': True
                    }
                    
                    optimized.append(merged_chunk)
                    i += 2  # Passer les deux chunks fusionnés
                    continue
            
            # Pas de fusion possible, garder le chunk tel quel
            optimized.append(current_chunk)
            i += 1
        
        return optimized
    
    def _merge_juridical_content(self, content1: Dict, content2: Dict) -> Dict:
        """Fusionne les informations de contenu juridique de deux chunks."""
        return {
            'has_articles': content1['has_articles'] or content2['has_articles'],
            'has_considerations': content1['has_considerations'] or content2['has_considerations'],
            'has_dispositif': content1['has_dispositif'] or content2['has_dispositif'],
            'has_references': content1['has_references'] or content2['has_references'],
            'has_dates': content1['has_dates'] or content2['has_dates'],
            'has_legal_terms': content1['has_legal_terms'] + content2['has_legal_terms']
        }
    
    def _detect_juridical_content(self, text: str) -> Dict:
        """
        Détecte le type de contenu juridique dans un chunk (version optimisée).
        """
        text_lower = text.lower()
        
        return {
            'has_articles': bool(re.search(r'\barticle\s+\d+', text_lower)),
            'has_considerations': any(marker in text_lower for marker in ['considérant', 'attendu']),
            'has_dispositif': 'par ces motifs' in text_lower or 'dispositif' in text_lower,
            'has_references': bool(re.search(r'(art\.|article)\s*\d+', text_lower)),
            'has_dates': bool(re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', text)),
            'has_legal_terms': sum(1 for marker in self.juridical_markers if marker in text_lower)
        }
    
    def analyze_chunks_with_ai_safe(self, chunks: List[Dict], prompt: str, 
                                   ai_function, progress_callback=None, **ai_params) -> List[Dict]:
        """
        Analyse chaque chunk avec l'IA en mode Cloudflare-safe.
        
        Args:
            chunks: Liste des chunks à analyser
            prompt: Prompt d'analyse pour l'IA
            ai_function: Fonction d'appel à l'IA
            progress_callback: Fonction de callback pour le progrès
            **ai_params: Paramètres pour l'IA
            
        Returns:
            Liste des chunks avec leurs analyses
        """
        print(f"CHUNK_ANALYZER_SAFE: Analyse de {len(chunks)} chunks (timeout {self.total_timeout}s)")
        
        self._start_time = time.time()
        self._cancelled = False
        analyzed_chunks = []
        
        # Traitement séquentiel optimisé (plus prévisible que parallèle)
        for i, chunk in enumerate(chunks, 1):
            # Vérification timeout global
            elapsed = time.time() - self._start_time
            if elapsed > self.total_timeout - 10:  # Marge sécurité 10s
                print(f"⚠️ Timeout global approché ({elapsed:.1f}s), arrêt analyses")
                break
            
            if self._cancelled:
                print(f"❌ Analyse annulée à l'étape {i}")
                break
            
            print(f"CHUNK_ANALYZER_SAFE: Analyse chunk {i}/{len(chunks)} (timeout individuel: {self.max_chunk_time}s)")
            
            # Callback de progrès
            if progress_callback:
                progress = i / len(chunks)
                progress_callback(progress * 0.8, f"Analyse chunk {i}/{len(chunks)}")  # 80% pour chunks
            
            # Préparer le prompt spécialisé pour l'analyse par chunk
            chunk_prompt = f"""ANALYSE RAPIDE PAR CHUNK - PARTIE {i}/{len(chunks)}

CONTEXTE: Analysez rapidement cette section d'un document juridique (traitement accéléré).

CONSIGNES ULTRA-RAPIDES:
1. Identifiez les éléments juridiques clés en 2-3 phrases max
2. Style télégraphique uniquement
3. Pas de développement, que l'essentiel
4. Format: "Moyen X: [résumé]"

EXTRAIT À ANALYSER:
{chunk['text']}"""

            try:
                # Appel à l'IA avec timeout strict par chunk
                start_chunk = time.time()
                
                analysis = ai_function(
                    text=chunk['text'],
                    prompt=chunk_prompt,
                    timeout=self.max_chunk_time,  # Timeout strict par chunk
                    **ai_params
                )
                
                chunk_duration = time.time() - start_chunk
                
                chunk_result = chunk.copy()
                chunk_result['analysis'] = analysis
                chunk_result['analysis_timestamp'] = datetime.now().isoformat()
                chunk_result['analysis_success'] = True
                chunk_result['processing_duration'] = chunk_duration
                
                analyzed_chunks.append(chunk_result)
                
                print(f"✅ Chunk {i} analysé en {chunk_duration:.1f}s")
                
                # Pause micro pour éviter surcharge
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Erreur analyse chunk {i}: {e}")
                chunk_result = chunk.copy()
                chunk_result['analysis'] = f"ERREUR: {str(e)}"
                chunk_result['analysis_success'] = False
                chunk_result['processing_duration'] = 0
                analyzed_chunks.append(chunk_result)
        
        total_elapsed = time.time() - self._start_time
        print(f"✅ Analyse chunks terminée en {total_elapsed:.1f}s ({len(analyzed_chunks)} réussis)")
        
        # Callback final
        if progress_callback:
            progress_callback(0.8, f"Chunks analysés: {len(analyzed_chunks)}")
        
        return analyzed_chunks
    
    def synthesize_analyses_fast(self, analyzed_chunks: List[Dict], 
                                synthesis_prompt: str, ai_function, 
                                progress_callback=None, **ai_params) -> str:
        """
        Synthétise rapidement les analyses de tous les chunks.
        
        Args:
            analyzed_chunks: Chunks avec leurs analyses
            synthesis_prompt: Prompt pour la synthèse
            ai_function: Fonction d'appel à l'IA
            progress_callback: Callback de progrès
            **ai_params: Paramètres pour l'IA
            
        Returns:
            Synthèse structurée
        """
        print("CHUNK_ANALYZER_SAFE: Synthèse rapide des analyses")
        
        if progress_callback:
            progress_callback(0.85, "Préparation synthèse...")
        
        # Construire le contenu pour la synthèse (condensé)
        synthesis_content = []
        successful_analyses = 0
        
        for chunk in analyzed_chunks:
            if chunk['analysis_success']:
                successful_analyses += 1
                # Version condensée pour la synthèse
                synthesis_content.append(f"""
CHUNK {chunk['id']} ({chunk['size']} chars):
{chunk['analysis'][:500]}{'...' if len(chunk['analysis']) > 500 else ''}
""")
            else:
                synthesis_content.append(f"CHUNK {chunk['id']}: ERREUR - {chunk['analysis']}")
        
        # Créer le prompt de synthèse optimisé
        full_synthesis_prompt = f"""SYNTHÈSE JURIDIQUE RAPIDE

MISSION: Synthèse express de {len(analyzed_chunks)} chunks ({successful_analyses} réussis).

CONSIGNES SYNTHÈSE RAPIDE:
1. Style télégraphique uniquement  
2. Maximum 3 paragraphes
3. Moyens principaux uniquement
4. Pas de redondances

PROMPT UTILISATEUR:
{synthesis_prompt}

ANALYSES À SYNTHÉTISER:
{''.join(synthesis_content[:10])}
{'...[autres chunks]...' if len(synthesis_content) > 10 else ''}

STRUCTURE ATTENDUE:
1. MOYENS PRINCIPAUX (résumé)
2. ARGUMENTS CLÉS (condensés)  
3. CONCLUSION (1 phrase)"""

        try:
            if progress_callback:
                progress_callback(0.9, "Génération synthèse...")
            
            synthesis_start = time.time()
            
            synthesis = ai_function(
                text="SYNTHÈSE DEMANDÉE",
                prompt=full_synthesis_prompt,
                timeout=min(25, self.total_timeout - (time.time() - self._start_time) - 5),  # Timeout adaptatif
                **ai_params
            )
            
            synthesis_duration = time.time() - synthesis_start
            print(f"✅ Synthèse générée en {synthesis_duration:.1f}s")
            
            if progress_callback:
                progress_callback(1.0, "Synthèse terminée")
            
            return synthesis
            
        except Exception as e:
            error_msg = f"ERREUR lors de la synthèse: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Synthèse de fallback
            fallback_synthesis = f"""SYNTHÈSE DE FALLBACK

Analyse par chunks effectuée sur {len(analyzed_chunks)} segments.
{successful_analyses} chunks analysés avec succès.

ÉLÉMENTS IDENTIFIÉS:
{chr(10).join([f"- Chunk {c['id']}: {'✅' if c['analysis_success'] else '❌'}" for c in analyzed_chunks[:5]])}

ERREUR SYNTHÈSE: {str(e)}
"""
            return fallback_synthesis
    
    def cancel_analysis(self):
        """Annule l'analyse en cours."""
        with self._lock:
            self._cancelled = True
        print("❌ Annulation de l'analyse demandée")
    
    def get_analysis_report_safe(self, chunks: List[Dict], 
                                analyzed_chunks: List[Dict]) -> str:
        """
        Génère un rapport détaillé optimisé sur l'analyse par chunks.
        """
        total_chars = sum(chunk['size'] for chunk in chunks)
        successful_analyses = sum(1 for chunk in analyzed_chunks if chunk['analysis_success'])
        total_time = time.time() - self._start_time if self._start_time else 0
        
        report = f"""
=== RAPPORT D'ANALYSE CLOUDFLARE-SAFE ===

PERFORMANCE:
- Document: {total_chars:,} caractères  
- Chunks: {len(chunks)} créés, {successful_analyses} analysés
- Durée totale: {total_time:.1f}s (limite: {self.total_timeout}s)
- Efficacité: {(successful_analyses/len(chunks)*100):.1f}%

CHUNKS TRAITÉS:
"""
        
        for chunk in analyzed_chunks[:10]:  # Limite affichage
            status = "✅" if chunk['analysis_success'] else "❌"
            duration = chunk.get('processing_duration', 0)
            juridical_info = chunk['contains_juridical']
            juridical_score = sum(juridical_info.values()) if isinstance(juridical_info, dict) else 0
            
            report += f"""
Chunk {chunk['id']:2d}: {status} ({duration:.1f}s)
  - Taille: {chunk['size']:,} chars | Juridique: {juridical_score} éléments
  - Estimation: {chunk.get('estimated_time', 0):.1f}s | Safe: {chunk.get('cloudflare_safe', True)}
"""
        
        if len(analyzed_chunks) > 10:
            report += f"\n... et {len(analyzed_chunks) - 10} autres chunks"
        
# =============================================================================
# ALIAS DE COMPATIBILITÉ
# =============================================================================

# Alias pour compatibilité avec le code existant
ChunkAnalyzer = CloudflareSafeChunkAnalyzer

# Fonction de compatibilité
def create_chunk_analyzer(chunk_size: int = 2000, overlap: int = 150) -> ChunkAnalyzer:
    """Crée un analyseur compatible avec l'ancien API."""
    return CloudflareSafeChunkAnalyzer(
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunk_time=30
    )

# =============================================================================
# FONCTIONS D'INTERFACE COMPATIBLES
# =============================================================================

def analyze_long_document_safe(text: str, prompt: str, ai_function, 
                              chunk_size: int = 2000, progress_callback=None, **ai_params) -> Dict:
    """
    Analyse un document long de manière sécurisée pour Cloudflare.
    
    Returns:
        Dict avec 'success', 'result', 'chunks_processed', 'duration'
    """
    start_time = time.time()
    
    try:
        # Créer l'analyseur
        analyzer = create_safe_chunk_analyzer(chunk_size=chunk_size)
        
        # Découper le texte
        if progress_callback:
            progress_callback(0.1, "Découpage en chunks...")
        
        chunks = analyzer.smart_chunk_text_safe(text)
        
        if progress_callback:
            progress_callback(0.2, f"{len(chunks)} chunks créés")
        
        # Analyser les chunks
        analyzed_chunks = analyzer.analyze_chunks_with_ai_safe(
            chunks, prompt, ai_function, progress_callback, **ai_params
        )
        
        # Synthétiser
        synthesis = analyzer.synthesize_analyses_fast(
            analyzed_chunks, prompt, ai_function, progress_callback, **ai_params
        )
        
        duration = time.time() - start_time
        
        return {
            'success': True,
            'result': synthesis,
            'chunks_processed': len(analyzed_chunks),
            'total_chunks': len(chunks),
            'duration': duration,
            'cloudflare_safe': duration < 80,
            'report': analyzer.get_analysis_report_safe(chunks, analyzed_chunks)
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'error': str(e),
            'duration': duration,
            'chunks_processed': 0,
            'total_chunks': 0
        }

# =============================================================================
# FONCTIONS D'INTERFACE SIMPLIFIÉES
# =============================================================================

def create_safe_chunk_analyzer(chunk_size: int = 2000, overlap: int = 150) -> CloudflareSafeChunkAnalyzer:
    """Crée un analyseur de chunks optimisé Cloudflare."""
    return CloudflareSafeChunkAnalyzer(
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunk_time=30  # 30s max par chunk
    )

def analyze_long_document_safe(text: str, prompt: str, ai_function, 
                              chunk_size: int = 2000, progress_callback=None, **ai_params) -> Dict:
    """
    Analyse un document long de manière sécurisée pour Cloudflare.
    
    Returns:
        Dict avec 'success', 'result', 'chunks_processed', 'duration'
    """
    start_time = time.time()
    
    try:
        # Créer l'analyseur
        analyzer = create_safe_chunk_analyzer(chunk_size=chunk_size)
        
        # Découper le texte
        if progress_callback:
            progress_callback(0.1, "Découpage en chunks...")
        
        chunks = analyzer.smart_chunk_text_safe(text)
        
        if progress_callback:
            progress_callback(0.2, f"{len(chunks)} chunks créés")
        
        # Analyser les chunks
        analyzed_chunks = analyzer.analyze_chunks_with_ai_safe(
            chunks, prompt, ai_function, progress_callback, **ai_params
        )
        
        # Synthétiser
        synthesis = analyzer.synthesize_analyses_fast(
            analyzed_chunks, prompt, ai_function, progress_callback, **ai_params
        )
        
        duration = time.time() - start_time
        
        return {
            'success': True,
            'result': synthesis,
            'chunks_processed': len(analyzed_chunks),
            'total_chunks': len(chunks),
            'duration': duration,
            'cloudflare_safe': duration < 80,
            'report': analyzer.get_analysis_report_safe(chunks, analyzed_chunks)
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'error': str(e),
            'duration': duration,
            'chunks_processed': 0,
            'total_chunks': 0
        }