#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Chunking Analysis - Découpage intelligent de documents
Version: 2.0 - Intégration corrigée
Date: 2025-09-16

Ce module corrige les problèmes d'intégration du chunking avec l'interface principale.
"""

import re
import time
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

@dataclass
class ChunkInfo:
    """Information sur un chunk de texte."""
    id: int
    content: str
    start_pos: int
    end_pos: int
    word_count: int
    char_count: int
    chunk_type: str  # 'paragraph', 'sentence', 'forced'
    overlap_with_previous: int
    overlap_with_next: int

@dataclass
class ChunkAnalysisResult:
    """Résultat de l'analyse d'un chunk."""
    chunk_id: int
    content: str
    processing_time: float
    tokens_used: int
    success: bool
    error_message: Optional[str] = None
    provider_used: str = ""
    model_used: str = ""

class DocumentChunker:
    """
    Classe pour découper intelligemment les documents longs.
    Corrige les problèmes d'intégration avec l'interface principale.
    """
    
    def __init__(self, max_chunk_size: int = 4000, overlap_size: int = 200):
        """
        Initialise le chunker.
        
        Args:
            max_chunk_size: Taille maximale d'un chunk en caractères
            overlap_size: Taille de l'overlap entre chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.chunk_history = []
        self._lock = threading.Lock()
        
        # Patterns pour détecter les structures de document
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
        self.section_pattern = re.compile(r'\n\s*(?:ARTICLE|CHAPITRE|SECTION|TITRE|\d+\.)\s', re.IGNORECASE)
        
    def analyze_document_structure(self, text: str) -> Dict:
        """Analyse la structure du document pour optimiser le découpage."""
        
        # Statistiques de base
        total_chars = len(text)
        total_words = len(text.split())
        total_lines = len(text.split('\n'))
        
        # Détection des structures
        paragraphs = self.paragraph_pattern.split(text)
        sentences = self.sentence_pattern.split(text)
        sections = self.section_pattern.findall(text)
        
        # Types de contenu détectés
        content_types = []
        if len(sections) > 3:
            content_types.append("document_structuré")
        if "article" in text.lower() or "clause" in text.lower():
            content_types.append("juridique")
        if len(paragraphs) > 10:
            content_types.append("texte_long")
        if re.search(r'\d+[.,]\d+', text):
            content_types.append("numérique")
        
        # Recommandation de stratégie
        if total_chars < self.max_chunk_size:
            strategy = "direct"
        elif len(sections) > 0:
            strategy = "by_section"
        elif len(paragraphs) > 5:
            strategy = "by_paragraph"
        else:
            strategy = "by_sentence"
        
        return {
            "total_chars": total_chars,
            "total_words": total_words,
            "total_lines": total_lines,
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "section_count": len(sections),
            "content_types": content_types,
            "recommended_strategy": strategy,
            "estimated_chunks": math.ceil(total_chars / self.max_chunk_size),
            "chunking_needed": total_chars > self.max_chunk_size
        }
    
    def smart_chunk_document(self, text: str, strategy: str = "auto") -> List[ChunkInfo]:
        """
        Découpe intelligemment un document selon la stratégie choisie.
        
        Args:
            text: Texte à découper
            strategy: Stratégie de découpage ('auto', 'by_section', 'by_paragraph', 'by_sentence', 'forced')
            
        Returns:
            Liste des chunks créés
        """
        
        if strategy == "auto":
            analysis = self.analyze_document_structure(text)
            strategy = analysis["recommended_strategy"]
        
        print(f"📄 Découpage avec stratégie: {strategy}")
        
        if strategy == "direct":
            return [ChunkInfo(
                id=1,
                content=text,
                start_pos=0,
                end_pos=len(text),
                word_count=len(text.split()),
                char_count=len(text),
                chunk_type="direct",
                overlap_with_previous=0,
                overlap_with_next=0
            )]
        
        elif strategy == "by_section":
            return self._chunk_by_sections(text)
        elif strategy == "by_paragraph":
            return self._chunk_by_paragraphs(text)
        elif strategy == "by_sentence":
            return self._chunk_by_sentences(text)
        else:  # forced
            return self._chunk_forced(text)
    
    def _chunk_by_sections(self, text: str) -> List[ChunkInfo]:
        """Découpe par sections (ARTICLE, CHAPITRE, etc.)."""
        
        # Trouve toutes les positions des sections
        section_positions = []
        for match in self.section_pattern.finditer(text):
            section_positions.append(match.start())
        
        if not section_positions:
            # Pas de sections détectées, fallback sur paragraphes
            return self._chunk_by_paragraphs(text)
        
        chunks = []
        section_positions.append(len(text))  # Ajouter la fin du document
        
        for i in range(len(section_positions) - 1):
            start_pos = section_positions[i]
            end_pos = section_positions[i + 1]
            section_text = text[start_pos:end_pos].strip()
            
            if len(section_text) <= self.max_chunk_size:
                # Section entière dans un chunk
                chunks.append(ChunkInfo(
                    id=len(chunks) + 1,
                    content=section_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    word_count=len(section_text.split()),
                    char_count=len(section_text),
                    chunk_type="section",
                    overlap_with_previous=0,
                    overlap_with_next=0
                ))
            else:
                # Section trop grande, découper en sous-chunks
                sub_chunks = self._split_large_section(section_text, start_pos)
                chunks.extend(sub_chunks)
        
        # Ajouter les overlaps
        self._add_overlaps(chunks, text)
        return chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[ChunkInfo]:
        """Découpe par paragraphes."""
        
        paragraphs = self.paragraph_pattern.split(text)
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Vérifier si ajouter ce paragraphe dépasse la limite
            potential_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if len(potential_chunk) <= self.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Sauvegarder le chunk actuel
                if current_chunk:
                    chunks.append(ChunkInfo(
                        id=len(chunks) + 1,
                        content=current_chunk,
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk),
                        word_count=len(current_chunk.split()),
                        char_count=len(current_chunk),
                        chunk_type="paragraph",
                        overlap_with_previous=0,
                        overlap_with_next=0
                    ))
                
                # Commencer un nouveau chunk
                current_chunk = para
                current_start = text.find(para, current_start + len(current_chunk) if current_chunk else 0)
        
        # Ajouter le dernier chunk
        if current_chunk:
            chunks.append(ChunkInfo(
                id=len(chunks) + 1,
                content=current_chunk,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                word_count=len(current_chunk.split()),
                char_count=len(current_chunk),
                chunk_type="paragraph",
                overlap_with_previous=0,
                overlap_with_next=0
            ))
        
        self._add_overlaps(chunks, text)
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[ChunkInfo]:
        """Découpe par phrases."""
        
        sentences = self.sentence_pattern.split(text)
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.max_chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(ChunkInfo(
                        id=len(chunks) + 1,
                        content=current_chunk,
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk),
                        word_count=len(current_chunk.split()),
                        char_count=len(current_chunk),
                        chunk_type="sentence",
                        overlap_with_previous=0,
                        overlap_with_next=0
                    ))
                
                current_chunk = sentence
                current_start = text.find(sentence, current_start + len(current_chunk) if current_chunk else 0)
        
        if current_chunk:
            chunks.append(ChunkInfo(
                id=len(chunks) + 1,
                content=current_chunk,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                word_count=len(current_chunk.split()),
                char_count=len(current_chunk),
                chunk_type="sentence",
                overlap_with_previous=0,
                overlap_with_next=0
            ))
        
        self._add_overlaps(chunks, text)
        return chunks
    
    def _chunk_forced(self, text: str) -> List[ChunkInfo]:
        """Découpe forcé par taille fixe."""
        
        chunks = []
        pos = 0
        
        while pos < len(text):
            end_pos = min(pos + self.max_chunk_size, len(text))
            
            # Essayer de couper sur un espace pour éviter de couper un mot
            if end_pos < len(text):
                last_space = text.rfind(' ', pos, end_pos)
                if last_space > pos:
                    end_pos = last_space
            
            chunk_text = text[pos:end_pos].strip()
            
            chunks.append(ChunkInfo(
                id=len(chunks) + 1,
                content=chunk_text,
                start_pos=pos,
                end_pos=end_pos,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                chunk_type="forced",
                overlap_with_previous=0,
                overlap_with_next=0
            ))
            
            pos = end_pos
        
        self._add_overlaps(chunks, text)
        return chunks
    
    def _split_large_section(self, section_text: str, base_start_pos: int) -> List[ChunkInfo]:
        """Divise une section trop grande en sous-chunks."""
        
        sub_chunks = []
        pos = 0
        
        while pos < len(section_text):
            end_pos = min(pos + self.max_chunk_size, len(section_text))
            
            # Chercher une coupure naturelle
            if end_pos < len(section_text):
                # Priorité aux paragraphes
                last_para = section_text.rfind('\n\n', pos, end_pos)
                if last_para > pos:
                    end_pos = last_para + 2
                else:
                    # Sinon sur une phrase
                    last_sentence = section_text.rfind('. ', pos, end_pos)
                    if last_sentence > pos:
                        end_pos = last_sentence + 2
            
            chunk_text = section_text[pos:end_pos].strip()
            
            sub_chunks.append(ChunkInfo(
                id=0,  # Sera réassigné
                content=chunk_text,
                start_pos=base_start_pos + pos,
                end_pos=base_start_pos + end_pos,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                chunk_type="sub_section",
                overlap_with_previous=0,
                overlap_with_next=0
            ))
            
            pos = end_pos
        
        return sub_chunks
    
    def _add_overlaps(self, chunks: List[ChunkInfo], original_text: str):
        """Ajoute les overlaps entre chunks consécutifs."""
        
        for i in range(len(chunks)):
            # Overlap avec le chunk précédent
            if i > 0:
                prev_chunk = chunks[i - 1]
                current_chunk = chunks[i]
                
                # Prendre les derniers mots du chunk précédent
                prev_words = prev_chunk.content.split()
                overlap_words = min(self.overlap_size // 10, len(prev_words) // 2)
                
                if overlap_words > 0:
                    overlap_text = " ".join(prev_words[-overlap_words:])
                    current_chunk.content = overlap_text + " " + current_chunk.content
                    current_chunk.overlap_with_previous = len(overlap_text)
                    prev_chunk.overlap_with_next = len(overlap_text)
            
            # Réassigner les IDs
            chunks[i].id = i + 1

class ChunkProcessor:
    """
    Classe pour traiter les chunks avec différents fournisseurs IA.
    Intégration corrigée avec le système principal.
    """
    
    def __init__(self, ai_provider_manager):
        """
        Initialise le processeur de chunks.
        
        Args:
            ai_provider_manager: Instance du gestionnaire de fournisseurs IA
        """
        self.manager = ai_provider_manager
        self.chunker = DocumentChunker()
        self.processing_history = []
        self._lock = threading.Lock()
    
    def process_document_with_chunking(self, text: str, provider: str, model: str,
                                      system_prompt: str, chunking_strategy: str = "auto",
                                      parallel_processing: bool = True, 
                                      synthesis_required: bool = True,
                                      **kwargs) -> Dict:
        """
        Traite un document long avec chunking intelligent.
        
        Args:
            text: Texte à traiter
            provider: Fournisseur IA à utiliser
            model: Modèle à utiliser
            system_prompt: Prompt système
            chunking_strategy: Stratégie de découpage
            parallel_processing: Traitement en parallèle des chunks
            synthesis_required: Synthèse finale requise
            **kwargs: Arguments supplémentaires pour le fournisseur
            
        Returns:
            Dict contenant les résultats du traitement
        """
        
        start_time = time.time()
        
        # Analyse de la structure du document
        doc_analysis = self.chunker.analyze_document_structure(text)
        print(f"📊 Analyse document: {doc_analysis['total_chars']} chars, {doc_analysis['estimated_chunks']} chunks estimés")
        
        # Vérification si chunking nécessaire
        if not doc_analysis["chunking_needed"]:
            print("✅ Document assez petit, traitement direct")
            return self._process_direct(text, provider, model, system_prompt, **kwargs)
        
        # Découpage du document
        chunks = self.chunker.smart_chunk_document(text, chunking_strategy)
        print(f"✂️ Document découpé en {len(chunks)} chunks")
        
        # Traitement des chunks
        if parallel_processing and len(chunks) > 1:
            chunk_results = self._process_chunks_parallel(chunks, provider, model, system_prompt, **kwargs)
        else:
            chunk_results = self._process_chunks_sequential(chunks, provider, model, system_prompt, **kwargs)
        
        # Synthèse des résultats
        if synthesis_required and len(chunk_results) > 1:
            final_content = self._synthesize_chunk_results(chunk_results, provider, model, system_prompt, **kwargs)
        else:
            final_content = self._concatenate_chunk_results(chunk_results)
        
        # Statistiques finales
        total_time = time.time() - start_time
        successful_chunks = sum(1 for r in chunk_results if r.success)
        
        result = {
            "status": "success" if successful_chunks > 0 else "failed",
            "processing_type": "chunking",
            "document_analysis": doc_analysis,
            "chunks_info": {
                "total_chunks": len(chunks),
                "successful_chunks": successful_chunks,
                "failed_chunks": len(chunks) - successful_chunks,
                "strategy_used": chunking_strategy,
                "parallel_processing": parallel_processing
            },
            "chunk_results": chunk_results,
            "final_content": final_content,
            "total_processing_time": total_time,
            "provider_used": provider,
            "model_used": model,
            "synthesis_applied": synthesis_required and len(chunk_results) > 1
        }
        
        # Sauvegarde dans l'historique
        with self._lock:
            self.processing_history.append(result)
        
        return result
    
    def _process_direct(self, text: str, provider: str, model: str, 
                       system_prompt: str, **kwargs) -> Dict:
        """Traitement direct sans chunking."""
        
        start_time = time.time()
        
        try:
            result_content = self.manager.generate(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                user_text=text,
                use_chunking=False,
                **kwargs
            )
            
            processing_time = time.time() - start_time
            success = not result_content.startswith("❌")
            
            return {
                "status": "success" if success else "failed",
                "processing_type": "direct",
                "final_content": result_content,
                "total_processing_time": processing_time,
                "provider_used": provider,
                "model_used": model,
                "chunks_info": {"total_chunks": 1, "successful_chunks": 1 if success else 0}
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "processing_type": "direct",
                "final_content": f"❌ Erreur traitement direct: {e}",
                "total_processing_time": time.time() - start_time,
                "provider_used": provider,
                "model_used": model
            }
    
    def _process_chunks_sequential(self, chunks: List[ChunkInfo], provider: str, 
                                  model: str, system_prompt: str, **kwargs) -> List[ChunkAnalysisResult]:
        """Traite les chunks séquentiellement."""
        
        results = []
        
        for i, chunk in enumerate(chunks):
            print(f"🔄 Traitement chunk {i+1}/{len(chunks)} ({chunk.char_count} chars)")
            
            # Prompt adapté pour le chunk
            chunk_prompt = self._create_chunk_prompt(system_prompt, chunk, i+1, len(chunks))
            
            start_time = time.time()
            
            try:
                result_content = self.manager.generate(
                    provider=provider,
                    model=model,
                    system_prompt=chunk_prompt,
                    user_text=chunk.content,
                    use_chunking=False,  # Déjà chunké
                    **kwargs
                )
                
                processing_time = time.time() - start_time
                success = not result_content.startswith("❌")
                
                results.append(ChunkAnalysisResult(
                    chunk_id=chunk.id,
                    content=result_content,
                    processing_time=processing_time,
                    tokens_used=len(chunk.content.split()) + len(result_content.split()),
                    success=success,
                    error_message=result_content if not success else None,
                    provider_used=provider,
                    model_used=model
                ))
                
                if success:
                    print(f"✅ Chunk {i+1} traité en {processing_time:.1f}s")
                else:
                    print(f"❌ Chunk {i+1} échoué: {result_content[:100]}...")
                
                # Pause entre chunks pour éviter la surcharge
                if i < len(chunks) - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                results.append(ChunkAnalysisResult(
                    chunk_id=chunk.id,
                    content="",
                    processing_time=time.time() - start_time,
                    tokens_used=0,
                    success=False,
                    error_message=str(e),
                    provider_used=provider,
                    model_used=model
                ))
                print(f"❌ Erreur chunk {i+1}: {e}")
        
        return results
    
    def _process_chunks_parallel(self, chunks: List[ChunkInfo], provider: str,
                                model: str, system_prompt: str, **kwargs) -> List[ChunkAnalysisResult]:
        """Traite les chunks en parallèle."""
        
        results = [None] * len(chunks)  # Pré-allouer pour maintenir l'ordre
        
        def process_single_chunk(chunk_index, chunk):
            """Fonction pour traiter un chunk individuel."""
            chunk = chunks[chunk_index]
            
            chunk_prompt = self._create_chunk_prompt(system_prompt, chunk, chunk_index+1, len(chunks))
            start_time = time.time()
            
            try:
                result_content = self.manager.generate(
                    provider=provider,
                    model=model,
                    system_prompt=chunk_prompt,
                    user_text=chunk.content,
                    use_chunking=False,
                    **kwargs
                )
                
                processing_time = time.time() - start_time
                success = not result_content.startswith("❌")
                
                return ChunkAnalysisResult(
                    chunk_id=chunk.id,
                    content=result_content,
                    processing_time=processing_time,
                    tokens_used=len(chunk.content.split()) + len(result_content.split()),
                    success=success,
                    error_message=result_content if not success else None,
                    provider_used=provider,
                    model_used=model
                )
                
            except Exception as e:
                return ChunkAnalysisResult(
                    chunk_id=chunk.id,
                    content="",
                    processing_time=time.time() - start_time,
                    tokens_used=0,
                    success=False,
                    error_message=str(e),
                    provider_used=provider,
                    model_used=model
                )
        
        # Traitement parallèle avec pool de threads
        max_workers = min(len(chunks), 4)  # Limite à 4 threads pour éviter la surcharge
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Soumettre tous les chunks
            future_to_index = {
                executor.submit(process_single_chunk, i, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collecter les résultats
            for future in as_completed(future_to_index):
                chunk_index = future_to_index[future]
                try:
                    result = future.result(timeout=120)  # Timeout de sécurité
                    results[chunk_index] = result
                    
                    if result.success:
                        print(f"✅ Chunk {chunk_index+1} traité en parallèle ({result.processing_time:.1f}s)")
                    else:
                        print(f"❌ Chunk {chunk_index+1} échoué en parallèle")
                        
                except Exception as e:
                    print(f"❌ Erreur chunk {chunk_index+1} en parallèle: {e}")
                    results[chunk_index] = ChunkAnalysisResult(
                        chunk_id=chunks[chunk_index].id,
                        content="",
                        processing_time=0,
                        tokens_used=0,
                        success=False,
                        error_message=str(e),
                        provider_used=provider,
                        model_used=model
                    )
        
        # Filtrer les résultats None (au cas où)
        return [r for r in results if r is not None]
    
    def _create_chunk_prompt(self, base_prompt: str, chunk: ChunkInfo, 
                           chunk_num: int, total_chunks: int) -> str:
        """Crée un prompt adapté pour un chunk spécifique."""
        
        chunk_context = f"""
CONTEXTE DE CHUNKING:
- Chunk {chunk_num}/{total_chunks}
- Position: caractères {chunk.start_pos}-{chunk.end_pos}
- Type de découpage: {chunk.chunk_type}
- Taille: {chunk.char_count} caractères, {chunk.word_count} mots
"""
        
        if chunk.overlap_with_previous > 0:
            chunk_context += f"- Overlap avec chunk précédent: {chunk.overlap_with_previous} caractères\n"
        
        if chunk_num == 1:
            chunk_context += "- PREMIER CHUNK: Établissez le contexte général\n"
        elif chunk_num == total_chunks:
            chunk_context += "- DERNIER CHUNK: Préparez une conclusion\n"
        else:
            chunk_context += "- CHUNK INTERMÉDIAIRE: Continuez l'analyse en cours\n"
        
        return f"{base_prompt}\n\n{chunk_context}\n\nINSTRUCTION: Analysez ce fragment en gardant à l'esprit qu'il fait partie d'un document plus large."
    
    def _synthesize_chunk_results(self, chunk_results: List[ChunkAnalysisResult],
                                 provider: str, model: str, system_prompt: str, **kwargs) -> str:
        """Synthétise les résultats de plusieurs chunks."""
        
        successful_results = [r for r in chunk_results if r.success]
        
        if not successful_results:
            return "❌ Aucun chunk n'a été traité avec succès pour la synthèse"
        
        if len(successful_results) == 1:
            return successful_results[0].content
        
        # Création du texte pour la synthèse
        synthesis_input = "SYNTHÈSE DE DOCUMENT DÉCOUPÉ\n\n"
        
        for result in successful_results:
            synthesis_input += f"=== CHUNK {result.chunk_id} ===\n"
            synthesis_input += f"Temps de traitement: {result.processing_time:.1f}s\n"
            synthesis_input += f"Contenu:\n{result.content}\n\n"
        
        # Prompt de synthèse
        synthesis_prompt = f"""
{system_prompt}

TÂCHE SPÉCIALE DE SYNTHÈSE:
Vous recevez les analyses de plusieurs fragments d'un même document.
Votre mission est de créer une synthèse cohérente et complète qui:

1. INTÈGRE tous les éléments pertinents de chaque fragment
2. MAINTIENT la logique et la continuité du document original
3. ÉVITE les répétitions entre les fragments
4. STRUCTURE l'information de manière claire et logique
5. FOURNIT une conclusion globale cohérente

Ne mentionnez pas les numéros de chunks dans votre réponse finale.
Créez un texte fluide comme si vous aviez analysé le document complet d'un coup.
"""
        
        try:
            print("🔄 Synthèse des résultats de chunks...")
            synthesis = self.manager.generate(
                provider=provider,
                model=model,
                system_prompt=synthesis_prompt,
                user_text=synthesis_input,
                use_chunking=False,
                temperature=kwargs.get('temperature', 0.3),  # Légèrement plus créatif pour la synthèse
                **{k: v for k, v in kwargs.items() if k != 'temperature'}
            )
            
            if synthesis.startswith("❌"):
                print("⚠️ Échec de la synthèse, concaténation des résultats")
                return self._concatenate_chunk_results(successful_results)
            
            print("✅ Synthèse terminée avec succès")
            return synthesis
            
        except Exception as e:
            print(f"❌ Erreur lors de la synthèse: {e}")
            return self._concatenate_chunk_results(successful_results)
    
    def _concatenate_chunk_results(self, chunk_results: List[ChunkAnalysisResult]) -> str:
        """Concatène simplement les résultats des chunks."""
        
        successful_results = [r for r in chunk_results if r.success]
        
        if not successful_results:
            return "❌ Aucun chunk traité avec succès"
        
        concatenated = "ANALYSE PAR FRAGMENTS\n\n"
        
        for result in successful_results:
            concatenated += f"--- Fragment {result.chunk_id} ---\n"
            concatenated += f"{result.content}\n\n"
        
        return concatenated
    
    def get_processing_stats(self) -> Dict:
        """Retourne les statistiques de traitement."""
        
        with self._lock:
            if not self.processing_history:
                return {"message": "Aucun traitement effectué"}
            
            total_processes = len(self.processing_history)
            successful_processes = sum(1 for p in self.processing_history if p["status"] == "success")
            
            chunking_processes = sum(1 for p in self.processing_history if p.get("processing_type") == "chunking")
            
            avg_time = sum(p.get("total_processing_time", 0) for p in self.processing_history) / total_processes
            
            total_chunks_processed = sum(
                p.get("chunks_info", {}).get("total_chunks", 0) 
                for p in self.processing_history
            )
            
            return {
                "total_processes": total_processes,
                "success_rate": f"{successful_processes}/{total_processes} ({successful_processes/total_processes*100:.1f}%)",
                "chunking_usage": f"{chunking_processes}/{total_processes} ({chunking_processes/total_processes*100:.1f}%)",
                "average_processing_time": f"{avg_time:.1f}s",
                "total_chunks_processed": total_chunks_processed,
                "average_chunks_per_document": f"{total_chunks_processed/total_processes:.1f}" if total_processes > 0 else "0"
            }

# =============================================================================
# FONCTIONS D'INTÉGRATION AVEC L'INTERFACE PRINCIPALE
# =============================================================================

def create_chunk_processor(ai_provider_manager):
    """Factory function pour créer un processeur de chunks."""
    return ChunkProcessor(ai_provider_manager)

def get_chunking_strategies() -> List[str]:
    """Retourne les stratégies de chunking disponibles."""
    return ["auto", "by_section", "by_paragraph", "by_sentence", "forced"]

def get_chunking_strategy_description(strategy: str) -> str:
    """Retourne la description d'une stratégie de chunking."""
    descriptions = {
        "auto": "🤖 Automatique - Détection intelligente de la meilleure stratégie",
        "by_section": "📋 Par sections - Découpe sur ARTICLE, CHAPITRE, etc.",
        "by_paragraph": "📝 Par paragraphes - Préserve la structure des paragraphes",
        "by_sentence": "💬 Par phrases - Découpe plus fin sur les phrases",
        "forced": "✂️ Forcé - Découpe par taille fixe (dernier recours)"
    }
    return descriptions.get(strategy, "Stratégie inconnue")

def estimate_chunking_benefit(text_length: int, model: str) -> Dict:
    """Estime les bénéfices du chunking pour un document."""
    
    # Estimation sans chunking
    from ai_providers import estimate_processing_time
    direct_estimation = estimate_processing_time(text_length, model, "normal")
    
    # Estimation avec chunking
    chunk_size = 4000
    estimated_chunks = max(1, text_length // chunk_size)
    chunk_time = estimate_processing_time(chunk_size, model, "normal")["estimated_time"]
    
    # Temps avec chunking (séquentiel + synthèse)
    chunking_time = (chunk_time * estimated_chunks) + (chunk_time * 0.5)  # +50% pour synthèse
    
    # Temps avec chunking parallèle
    parallel_time = chunk_time + (chunk_time * 0.5)  # Chunks en parallèle + synthèse
    
    return {
        "text_length": text_length,
        "estimated_chunks": estimated_chunks,
        "direct_processing": {
            "time": direct_estimation["estimated_time"],
            "cloudflare_safe": direct_estimation["cloudflare_safe"],
            "strategy": direct_estimation["strategy"]
        },
        "sequential_chunking": {
            "time": chunking_time,
            "recommended": chunking_time < direct_estimation["estimated_time"] * 0.8
        },
        "parallel_chunking": {
            "time": parallel_time,
            "recommended": parallel_time < direct_estimation["estimated_time"] * 0.6
        },
        "recommendation": "chunking" if not direct_estimation["cloudflare_safe"] else "direct"
    }

# =============================================================================
# EXEMPLE D'UTILISATION COMPLÈTE
# =============================================================================

def example_chunking_usage():
    """Exemple d'utilisation complète du système de chunking."""
    
    # Supposons que vous avez déjà un AIProviderManager
    from ai_providers import AIProviderManager
    
    manager = AIProviderManager()
    processor = ChunkProcessor(manager)
    
    # Document exemple (long)
    long_document = """
    CONTRAT DE PRESTATIONS DE SERVICES
    
    ARTICLE 1 - OBJET
    Le présent contrat a pour objet la définition des conditions...
    
    ARTICLE 2 - OBLIGATIONS DU PRESTATAIRE
    Le prestataire s'engage à...
    
    [... document très long avec de nombreux articles ...]
    """ * 50  # Simuler un document très long
    
    # Traitement avec chunking
    result = processor.process_document_with_chunking(
        text=long_document,
        provider="ollama_local",
        model="mistral:7b-instruct",
        system_prompt="Tu es un expert juridique. Analyse ce contrat en détail.",
        chunking_strategy="by_section",
        parallel_processing=True,
        synthesis_required=True,
        temperature=0.2
    )
    
    print(f"Statut: {result['status']}")
    print(f"Type de traitement: {result['processing_type']}")
    print(f"Chunks traités: {result['chunks_info']['successful_chunks']}/{result['chunks_info']['total_chunks']}")
    print(f"Temps total: {result['total_processing_time']:.1f}s")
    print(f"Synthèse appliquée: {result['synthesis_applied']}")
    print(f"Contenu final: {result['final_content'][:300]}...")
    
    # Statistiques
    stats = processor.get_processing_stats()
    print(f"Statistiques: {stats}")

if __name__ == "__main__":
    example_chunking_usage()
                