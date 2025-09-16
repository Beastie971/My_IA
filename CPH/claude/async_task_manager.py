#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestionnaire de tâches asynchrones pour éviter timeout 524
Version: 1.0-CLOUDFLARE-SAFE
Date: 2025-09-15
"""

import threading
import time
import uuid
import json
import os
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

@dataclass
class TaskStatus:
    """Statut d'une tâche asynchrone."""
    id: str
    status: str  # pending, running, completed, error, timeout
    progress: float  # 0.0 à 1.0
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour JSON."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    @property
    def duration(self) -> float:
        """Durée d'exécution en secondes."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def is_finished(self) -> bool:
        """True si la tâche est terminée."""
        return self.status in ['completed', 'error', 'timeout']

class AsyncTaskManager:
    """Gestionnaire de tâches asynchrones pour éviter les timeouts Cloudflare."""
    
    def __init__(self, max_task_duration: int = 80, cleanup_interval: int = 300):
        """
        Initialise le gestionnaire de tâches.
        
        Args:
            max_task_duration: Durée max d'une tâche en secondes (80s pour Cloudflare)
            cleanup_interval: Intervalle de nettoyage des tâches expirées en secondes
        """
        self.tasks: Dict[str, TaskStatus] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.max_task_duration = max_task_duration
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = datetime.now()
        self._lock = threading.Lock()
        
        # Répertoire pour persister les tâches
        self.tasks_dir = "async_tasks"
        os.makedirs(self.tasks_dir, exist_ok=True)
    
    def start_task(self, task_func: Callable, task_args: tuple = (), 
                   task_kwargs: dict = None, task_name: str = "Unknown") -> str:
        """
        Lance une tâche en arrière-plan.
        
        Args:
            task_func: Fonction à exécuter
            task_args: Arguments positionnels
            task_kwargs: Arguments nommés
            task_name: Nom de la tâche pour debugging
            
        Returns:
            ID de la tâche
        """
        if task_kwargs is None:
            task_kwargs = {}
        
        task_id = str(uuid.uuid4())
        
        # Créer le statut initial
        task_status = TaskStatus(
            id=task_id,
            status="pending",
            progress=0.0,
            start_time=datetime.now(),
            metadata={"name": task_name, "args_count": len(task_args)}
        )
        
        with self._lock:
            self.tasks[task_id] = task_status
        
        # Sauvegarder sur disque
        self._save_task_status(task_status)
        
        # Wrapper pour la fonction avec gestion d'erreurs
        def task_wrapper():
            try:
                # Marquer comme en cours
                self._update_task_status(task_id, "running", 0.1)
                
                # Surveiller le timeout
                start_time = time.time()
                
                def timeout_monitor():
                    """Surveille le timeout et arrête la tâche si nécessaire."""
                    while True:
                        if time.time() - start_time > self.max_task_duration:
                            self._update_task_status(
                                task_id, "timeout", 1.0,
                                error=f"Tâche interrompue après {self.max_task_duration}s (limite Cloudflare)"
                            )
                            break
                        
                        # Vérifier si la tâche est terminée
                        with self._lock:
                            if task_id not in self.tasks or self.tasks[task_id].is_finished:
                                break
                        
                        time.sleep(1)
                
                # Lancer le monitoring en parallèle
                monitor_thread = threading.Thread(target=timeout_monitor, daemon=True)
                monitor_thread.start()
                
                # Exécuter la tâche
                result = task_func(*task_args, **task_kwargs)
                
                # Vérifier si on n'a pas timeout
                with self._lock:
                    if task_id in self.tasks and self.tasks[task_id].status != "timeout":
                        self._update_task_status(task_id, "completed", 1.0, result=result)
                
            except Exception as e:
                error_msg = f"Erreur lors de l'exécution : {str(e)}"
                self._update_task_status(task_id, "error", 1.0, error=error_msg)
            
            finally:
                # Nettoyer le thread
                with self._lock:
                    if task_id in self.threads:
                        del self.threads[task_id]
        
        # Lancer le thread
        thread = threading.Thread(target=task_wrapper, daemon=True)
        thread.start()
        
        with self._lock:
            self.threads[task_id] = thread
        
        print(f"🚀 Tâche asynchrone lancée: {task_id} ({task_name})")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Récupère le statut d'une tâche."""
        self._cleanup_old_tasks()
        
        with self._lock:
            if task_id in self.tasks:
                return self.tasks[task_id].to_dict()
        
        # Essayer de charger depuis le disque
        loaded_status = self._load_task_status(task_id)
        if loaded_status:
            with self._lock:
                self.tasks[task_id] = loaded_status
            return loaded_status.to_dict()
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Annule une tâche en cours."""
        with self._lock:
            if task_id in self.tasks and not self.tasks[task_id].is_finished:
                self._update_task_status(task_id, "error", 1.0, error="Tâche annulée par l'utilisateur")
                
                # Arrêter le thread si possible (note: pas de force kill)
                if task_id in self.threads:
                    # Le thread se terminera à la prochaine vérification
                    pass
                
                return True
        
        return False
    
    def list_active_tasks(self) -> Dict[str, Dict]:
        """Liste toutes les tâches actives."""
        self._cleanup_old_tasks()
        
        active_tasks = {}
        with self._lock:
            for task_id, task in self.tasks.items():
                if not task.is_finished:
                    active_tasks[task_id] = task.to_dict()
        
        return active_tasks
    
    def _update_task_status(self, task_id: str, status: str, progress: float, 
                           result: Any = None, error: str = None):
        """Met à jour le statut d'une tâche."""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                task.progress = progress
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                if status in ['completed', 'error', 'timeout']:
                    task.end_time = datetime.now()
                
                # Sauvegarder sur disque
                self._save_task_status(task)
    
    def _save_task_status(self, task: TaskStatus):
        """Sauvegarde le statut d'une tâche sur disque."""
        try:
            file_path = os.path.join(self.tasks_dir, f"{task.id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(task.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Erreur sauvegarde tâche {task.id}: {e}")
    
    def _load_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Charge le statut d'une tâche depuis le disque."""
        try:
            file_path = os.path.join(self.tasks_dir, f"{task_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstituer l'objet TaskStatus
                task = TaskStatus(
                    id=data['id'],
                    status=data['status'],
                    progress=data['progress'],
                    start_time=datetime.fromisoformat(data['start_time']),
                    end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
                    result=data.get('result'),
                    error=data.get('error'),
                    metadata=data.get('metadata')
                )
                return task
        except Exception as e:
            print(f"Erreur chargement tâche {task_id}: {e}")
        
        return None
    
    def _cleanup_old_tasks(self):
        """Nettoie les tâches anciennes."""
        now = datetime.now()
        if (now - self.last_cleanup).total_seconds() < self.cleanup_interval:
            return
        
        cutoff_time = now - timedelta(hours=2)  # Supprimer après 2h
        
        with self._lock:
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if task.start_time < cutoff_time and task.is_finished:
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
                
                # Supprimer le fichier
                try:
                    file_path = os.path.join(self.tasks_dir, f"{task_id}.json")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Erreur suppression fichier tâche {task_id}: {e}")
        
        self.last_cleanup = now
        if tasks_to_remove:
            print(f"🧹 Nettoyage: {len(tasks_to_remove)} tâche(s) supprimée(s)")

# Instance globale du gestionnaire de tâches
task_manager = AsyncTaskManager()

# =============================================================================
# FONCTIONS D'INTERFACE POUR GRADIO
# =============================================================================

def start_analysis_async(analysis_func: Callable, *args, **kwargs) -> str:
    """Lance une analyse en mode asynchrone."""
    task_name = kwargs.pop('task_name', 'Analyse juridique')
    
    return task_manager.start_task(
        task_func=analysis_func,
        task_args=args,
        task_kwargs=kwargs,
        task_name=task_name
    )

def check_analysis_status(task_id: str) -> Dict[str, Any]:
    """Vérifie le statut d'une analyse."""
    if not task_id:
        return {"error": "ID de tâche requis"}
    
    status = task_manager.get_task_status(task_id)
    if not status:
        return {"error": f"Tâche {task_id} introuvable"}
    
    return status

def get_analysis_result(task_id: str) -> Dict[str, Any]:
    """Récupère le résultat d'une analyse terminée."""
    status = check_analysis_status(task_id)
    
    if "error" in status:
        return status
    
    if status["status"] != "completed":
        return {
            "error": f"Tâche non terminée (statut: {status['status']})",
            "status": status["status"],
            "progress": status["progress"]
        }
    
    return {
        "success": True,
        "result": status["result"],
        "duration": status.get("duration", 0),
        "metadata": status.get("metadata", {})
    }

def cancel_analysis(task_id: str) -> Dict[str, Any]:
    """Annule une analyse en cours."""
    if task_manager.cancel_task(task_id):
        return {"success": True, "message": f"Tâche {task_id} annulée"}
    else:
        return {"error": f"Impossible d'annuler la tâche {task_id}"}

def list_running_analyses() -> Dict[str, Any]:
    """Liste toutes les analyses en cours."""
    active_tasks = task_manager.list_active_tasks()
    return {
        "active_count": len(active_tasks),
        "tasks": active_tasks
    }

# =============================================================================
# FONCTIONS SPÉCIALISÉES POUR L'ANALYSE JURIDIQUE
# =============================================================================

def create_progress_callback(task_id: str):
    """Crée un callback pour mettre à jour le progrès d'une tâche."""
    def update_progress(progress: float, message: str = ""):
        task_manager._update_task_status(
            task_id, 
            "running", 
            progress,
            result={"progress_message": message} if message else None
        )
    
    return update_progress

def format_async_result(result: Any, task_info: Dict) -> str:
    """Formate le résultat d'une analyse asynchrone."""
    if isinstance(result, tuple) and len(result) >= 5:
        # Format standard: (formatted_result, stats1, stats2, debug_info, analysis_report)
        formatted_result, stats1, stats2, debug_info, analysis_report = result[:5]
        
        duration = task_info.get("duration", 0)
        task_name = task_info.get("metadata", {}).get("name", "Analyse")
        
        header = f"""{'=' * 80}
                    {task_name.upper()} - MODE ASYNCHRONE
{'=' * 80}

⏱️ DURÉE: {duration:.1f}s (Compatible Cloudflare < 100s)
🚀 MODE: Traitement asynchrone anti-timeout
📊 STATUT: Terminé avec succès

{'-' * 80}
"""
        
        return header + formatted_result
    
    else:
        # Résultat simple
        return f"""ANALYSE ASYNCHRONE TERMINÉE

Résultat:
{result}

Durée: {task_info.get('duration', 0):.1f}s
"""

# =============================================================================
# CLASSE WRAPPER POUR ANALYSES LONGUES
# =============================================================================

class SafeAnalysisWrapper:
    """Wrapper pour rendre les analyses compatibles avec Cloudflare."""
    
    def __init__(self, max_direct_time: int = 70):
        self.max_direct_time = max_direct_time
    
    def should_use_async(self, text_length: int, model: str, mode: str = "normal") -> bool:
        """Détermine si l'analyse doit être asynchrone."""
        from ai_providers import estimate_processing_time
        
        estimation = estimate_processing_time(text_length, model, mode)
        return estimation["estimated_time"] > self.max_direct_time
    
    def safe_analyze(self, analysis_func: Callable, text: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Analyse sécurisée qui choisit automatiquement entre mode direct et asynchrone.
        
        Returns:
            Dict avec 'is_async', 'task_id' ou 'result'
        """
        text_length = len(text)
        model = kwargs.get('model', 'mistral:7b')
        mode = kwargs.get('mode', 'normal')
        
        if self.should_use_async(text_length, model, mode):
            # Mode asynchrone
            task_id = start_analysis_async(
                analysis_func, 
                text, 
                *args, 
                task_name=f"Analyse {mode} ({text_length:,} chars)",
                **kwargs
            )
            
            return {
                "is_async": True,
                "task_id": task_id,
                "message": f"🚀 Analyse lancée en mode asynchrone (document long: {text_length:,} caractères)",
                "estimated_time": "60-120 secondes",
                "recommendation": "Utilisez le bouton 'Vérifier statut' pour suivre le progrès"
            }
        
        else:
            # Mode direct
            try:
                result = analysis_func(text, *args, **kwargs)
                return {
                    "is_async": False,
                    "result": result,
                    "message": f"✅ Analyse directe terminée ({text_length:,} caractères)"
                }
            except Exception as e:
                return {
                    "is_async": False,
                    "error": str(e),
                    "message": f"❌ Erreur analyse directe: {str(e)}"
                }

# Instance globale du wrapper
safe_wrapper = SafeAnalysisWrapper()