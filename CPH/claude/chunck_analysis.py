#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'analyse par chunks pour documents juridiques longs
Version: 1.0
Date: 2025-09-11
Fonctionnalités: Découpage intelligent, analyse par chunks, synthèse structurée
"""

import re
import json
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime

class ChunkAnalyzer:
    """Analyseur de documents longs par chunks avec synthèse juridique."""
    
    def __init__(self, chunk_size: int = 8000, overlap: int = 200):
        """
        Initialise l'analyseur de chunks.
        
        Args:
            chunk_size: Taille maximale d'un chunk en caractères
            overlap: Chevauchement entre chunks pour préserver le contexte
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.juridical_markers = [
            "article", "alinéa", "paragraphe", "considérant", "attendu",
            "par ces motifs", "dispositif", "en conséquence", "statuant",
            "vu", "considérant que", "attendu que", "il s'ensuit",
            "partant", "dès lors", "en effet", "cependant", "toutefois",
            "néanmoins", "or", "mais", "donc", "ainsi", "par ailleurs"
        ]
    
    def smart_chunk_text(self, text: str, preserve_structure: bool = True) -> List[Dict]:
        """
        Découpe intelligemment le texte en chunks en préservant la structure juridique.
        
        Args:
            text: Texte à découper
            preserve_structure: Si True, évite de couper au milieu des sections juridiques
            
        Returns:
            Liste de dictionnaires contenant les chunks avec métadonnées
        """
        print(f"CHUNK_ANALYZER: Découpage de {len(text)} caractères")
        
        if len(text) <= self.chunk_size:
            return [{
                'id': 1,
                'text': text,
                'start_pos': 0,
                'end_pos': len(text),
                'size': len(text),
                'overlap_prev': 0,
                'overlap_next': 0,
                'contains_juridical': self._detect_juridical_content(text)
            }]
        
        chunks = []
        current_pos = 0
        chunk_id = 1
        
        while current_pos < len(text):
            # Calculer la fin du chunk
            end_pos = min(current_pos + self.chunk_size, len(text))
            
            # Si on n'est pas à la fin du texte, essayer de trouver un point de coupe intelligent
            if end_pos < len(text) and preserve_structure:
                end_pos = self._find_smart_break(text, current_pos, end_pos)
            
            # Extraire le chunk
            chunk_text = text[current_pos:end_pos]
            
            # Ajouter le chevauchement avec le chunk précédent si nécessaire
            overlap_prev = 0
            if chunk_id > 1 and current_pos > 0:
                overlap_start = max(0, current_pos - self.overlap)
                overlap_text = text[overlap_start:current_pos]
                chunk_text = overlap_text + chunk_text
                overlap_prev = len(overlap_text)
            
            # Calculer le chevauchement avec le chunk suivant
            overlap_next = 0
            if end_pos < len(text):
                overlap_end = min(len(text), end_pos + self.overlap)
                next_overlap = text[end_pos:overlap_end]
                chunk_text += next_overlap
                overlap_next = len(next_overlap)
            
            chunk_info = {
                'id': chunk_id,
                'text': chunk_text,
                'start_pos': current_pos,
                'end_pos': end_pos,
                'size': len(chunk_text),
                'overlap_prev': overlap_prev,
                'overlap_next': overlap_next,
                'contains_juridical': self._detect_juridical_content(chunk_text)
            }
            
            chunks.append(chunk_info)
            
            # Passer au chunk suivant
            current_pos = end_pos
            chunk_id += 1
        
        print(f"CHUNK_ANALYZER: {len(chunks)} chunks créés")
        return chunks
    
    def _find_smart_break(self, text: str, start: int, proposed_end: int) -> int:
        """
        Trouve un point de coupe intelligent pour préserver la structure juridique.
        
        Args:
            text: Texte complet
            start: Position de début du chunk
            proposed_end: Position de fin proposée
            
        Returns:
            Position de fin optimisée
        """
        # Zone de recherche pour le point de coupe (500 caractères avant la fin proposée)
        search_start = max(start + self.chunk_size - 500, start)
        search_text = text[search_start:proposed_end + 200]
        
        # Priorités pour les points de coupe (du meilleur au moins bon)
        break_patterns = [
            r'\n\n(?=[A-Z])',  # Double saut de ligne suivi d'une majuscule
            r'\n(?=Article|Considérant|Attendu|Par ces motifs)',  # Début d'article juridique
            r'\.[\s\n]+(?=[A-Z])',  # Fin de phrase suivie d'une majuscule
            r'\n(?=[0-9]+[\.\)])',  # Début de numérotation
            r'[\.!?][\s\n]+',  # Fin de phrase
            r'\n',  # Simple saut de ligne
        ]
        
        for pattern in break_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Prendre le dernier match dans la zone de recherche
                best_match = matches[-1]
                return search_start + best_match.end()
        
        # Si aucun point de coupe intelligent trouvé, utiliser la position proposée
        return proposed_end
    
    def _detect_juridical_content(self, text: str) -> Dict:
        """
        Détecte le type de contenu juridique dans un chunk.
        
        Args:
            text: Texte du chunk
            
        Returns:
            Dictionnaire avec les types de contenu détectés
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
    
    def analyze_chunks_with_ai(self, chunks: List[Dict], prompt: str, 
                             ai_function, **ai_params) -> List[Dict]:
        """
        Analyse chaque chunk avec l'IA et retourne les résultats.
        
        Args:
            chunks: Liste des chunks à analyser
            prompt: Prompt d'analyse pour l'IA
            ai_function: Fonction d'appel à l'IA
            **ai_params: Paramètres pour l'IA
            
        Returns:
            Liste des chunks avec leurs analyses
        """
        print(f"CHUNK_ANALYZER: Analyse de {len(chunks)} chunks avec l'IA")
        
        analyzed_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"CHUNK_ANALYZER: Analyse du chunk {i}/{len(chunks)}")
            
            # Préparer le prompt spécialisé pour l'analyse par chunk
            chunk_prompt = f"""ANALYSE PAR CHUNK - PARTIE {i}/{len(chunks)}

CONTEXTE: Vous analysez la partie {i} d'un document juridique long découpé en {len(chunks)} segments.

CONTENU DU CHUNK {i}:
Position: caractères {chunk['start_pos']} à {chunk['end_pos']}
Taille: {chunk['size']} caractères
Contenu juridique détecté: {chunk['contains_juridical']}

INSTRUCTIONS D'ANALYSE:
{prompt}

CONSIGNES SPÉCIALES POUR LES CHUNKS:
1. Analysez UNIQUEMENT le contenu de ce chunk
2. Identifiez les éléments juridiques clés de cette section
3. Notez si des éléments semblent incomplets (début/fin de phrase coupée)
4. Structurez votre réponse pour faciliter la synthèse finale

TEXTE À ANALYSER:
{chunk['text']}"""

            try:
                # Appel à l'IA pour analyser ce chunk
                analysis = ai_function(
                    text=chunk['text'],
                    prompt=chunk_prompt,
                    **ai_params
                )
                
                chunk_result = chunk.copy()
                chunk_result['analysis'] = analysis
                chunk_result['analysis_timestamp'] = datetime.now().isoformat()
                chunk_result['analysis_success'] = True
                
                analyzed_chunks.append(chunk_result)
                
                # Pause entre les analyses pour éviter la surcharge
                time.sleep(1)
                
            except Exception as e:
                print(f"CHUNK_ANALYZER: Erreur analyse chunk {i}: {e}")
                chunk_result = chunk.copy()
                chunk_result['analysis'] = f"ERREUR: {str(e)}"
                chunk_result['analysis_success'] = False
                analyzed_chunks.append(chunk_result)
        
        return analyzed_chunks
    
    def synthesize_analyses(self, analyzed_chunks: List[Dict], 
                          synthesis_prompt: str, ai_function, **ai_params) -> str:
        """
        Synthétise les analyses de tous les chunks en un rapport structuré.
        
        Args:
            analyzed_chunks: Chunks avec leurs analyses
            synthesis_prompt: Prompt pour la synthèse
            ai_function: Fonction d'appel à l'IA
            **ai_params: Paramètres pour l'IA
            
        Returns:
            Synthèse structurée
        """
        print("CHUNK_ANALYZER: Synthèse des analyses")
        
        # Construire le contenu pour la synthèse
        synthesis_content = []
        
        for chunk in analyzed_chunks:
            if chunk['analysis_success']:
                synthesis_content.append(f"""
=== ANALYSE CHUNK {chunk['id']} ===
Position: {chunk['start_pos']}-{chunk['end_pos']} ({chunk['size']} caractères)
Contenu juridique: {chunk['contains_juridical']}

ANALYSE:
{chunk['analysis']}
""")
            else:
                synthesis_content.append(f"""
=== CHUNK {chunk['id']} - ERREUR ===
Position: {chunk['start_pos']}-{chunk['end_pos']}
Erreur: {chunk['analysis']}
""")
        
        # Créer le prompt de synthèse
        full_synthesis_prompt = f"""SYNTHÈSE JURIDIQUE STRUCTURÉE

MISSION: Créez une synthèse juridique structurée à partir des analyses de {len(analyzed_chunks)} chunks d'un document juridique long.

INSTRUCTIONS DE SYNTHÈSE:
{synthesis_prompt}

STRUCTURE ATTENDUE:
1. RÉSUMÉ EXÉCUTIF
2. ÉLÉMENTS JURIDIQUES PRINCIPAUX
3. CHRONOLOGIE/PROCÉDURE
4. POINTS CLÉS PAR THÉMATIQUE
5. CONCLUSIONS ET RECOMMANDATIONS

ANALYSES À SYNTHÉTISER:
{''.join(synthesis_content)}

CONSIGNES:
- Consolidez les informations des différents chunks
- Éliminez les redondances
- Structurez de manière logique et juridique
- Identifiez les liens entre les sections
- Proposez une analyse globale cohérente"""

        try:
            synthesis = ai_function(
                text="SYNTHÈSE DEMANDÉE",
                prompt=full_synthesis_prompt,
                **ai_params
            )
            return synthesis
        except Exception as e:
            return f"ERREUR lors de la synthèse: {str(e)}"
    
    def get_analysis_report(self, chunks: List[Dict], 
                          analyzed_chunks: List[Dict]) -> str:
        """
        Génère un rapport détaillé sur l'analyse par chunks.
        
        Args:
            chunks: Chunks originaux
            analyzed_chunks: Chunks analysés
            
        Returns:
            Rapport détaillé
        """
        total_chars = sum(chunk['size'] for chunk in chunks)
        successful_analyses = sum(1 for chunk in analyzed_chunks if chunk['analysis_success'])
        
        report = f"""
=== RAPPORT D'ANALYSE PAR CHUNKS ===

STATISTIQUES GÉNÉRALES:
- Document original: {total_chars:,} caractères
- Nombre de chunks: {len(chunks)}
- Analyses réussies: {successful_analyses}/{len(analyzed_chunks)}
- Taille moyenne par chunk: {total_chars // len(chunks):,} caractères

DÉTAIL DES CHUNKS:
"""
        
        for chunk in analyzed_chunks:
            status = "✅ SUCCÈS" if chunk['analysis_success'] else "❌ ÉCHEC"
            juridical_info = chunk['contains_juridical']
            juridical_score = sum(juridical_info.values()) if isinstance(juridical_info, dict) else 0
            
            report += f"""
Chunk {chunk['id']:2d}: {status}
  - Position: {chunk['start_pos']:,} à {chunk['end_pos']:,} ({chunk['size']:,} caractères)
  - Contenu juridique: {juridical_score} éléments détectés
  - Chevauchements: {chunk['overlap_prev']} | {chunk['overlap_next']}
"""
        
        return report
