#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration GPU Ollama - Correction des problèmes de sollicitation GPU
Version: 1.0
Date: 2025-09-16

Ce module corrige les problèmes de configuration GPU pour Ollama.
"""

import json
import requests
import subprocess
import platform
import time
import psutil
import GPUtil
from typing import Dict, List, Optional, Tuple
import os

class OllamaGPUConfigurator:
    """
    Configurateur pour optimiser l'utilisation GPU avec Ollama.
    Corrige les problèmes de non-sollicitation du GPU.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialise le configurateur GPU.
        
        Args:
            ollama_url: URL du serveur Ollama
        """
        self.ollama_url = ollama_url
        self.system_info = self._detect_system_info()
        self.gpu_info = self._detect_gpu_info()
        
    def _detect_system_info(self) -> Dict:
        """Détecte les informations système."""
        
        return {
            "os": platform.system(),
            "architecture": platform.machine(),
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        }
    
    def _detect_gpu_info(self) -> Dict:
        """Détecte les informations GPU."""
        
        gpu_info = {
            "gpus_available": False,
            "gpu_count": 0,
            "gpus": [],
            "cuda_available": False,
            "rocm_available": False,
            "metal_available": False
        }
        
        try:
            # Détection des GPUs avec GPUtil
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu_info["gpus_available"] = True
                gpu_info["gpu_count"] = len(gpus)
                
                for gpu in gpus:
                    gpu_info["gpus"].append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total_mb": gpu.memoryTotal,
                        "memory_free_mb": gpu.memoryFree,
                        "memory_used_mb": gpu.memoryUsed,
                        "utilization": gpu.load * 100,
                        "temperature": gpu.temperature
                    })
            
            # Détection CUDA
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                gpu_info["cuda_available"] = result.returncode == 0
            except:
                pass
            
            # Détection ROCm (AMD)
            try:
                result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
                gpu_info["rocm_available"] = result.returncode == 0
            except:
                pass
            
            # Détection Metal (macOS)
            if platform.system() == "Darwin":
                try:
                    # Vérifier si Metal est disponible (approximatif)
                    result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                          capture_output=True, text=True, timeout=10)
                    gpu_info["metal_available"] = "Metal" in result.stdout
                except:
                    pass
        
        except Exception as e:
            print(f"⚠️ Erreur détection GPU: {e}")
        
        return gpu_info
    
    def diagnose_gpu_issues(self) -> Dict:
        """Diagnostique les problèmes de configuration GPU."""
        
        issues = []
        recommendations = []
        severity = "info"
        
        # Vérification 1: Présence de GPU
        if not self.gpu_info["gpus_available"]:
            issues.append("Aucun GPU détecté sur le système")
            recommendations.append("Vérifiez l'installation des pilotes GPU")
            severity = "warning"
        
        # Vérification 2: Drivers CUDA/ROCm
        if self.gpu_info["gpus_available"]:
            nvidia_gpus = [gpu for gpu in self.gpu_info["gpus"] if "nvidia" in gpu["name"].lower()]
            amd_gpus = [gpu for gpu in self.gpu_info["gpus"] if "amd" in gpu["name"].lower() or "radeon" in gpu["name"].lower()]
            
            if nvidia_gpus and not self.gpu_info["cuda_available"]:
                issues.append("GPU NVIDIA détecté mais CUDA non disponible")
                recommendations.append("Installez les drivers NVIDIA et CUDA Toolkit")
                severity = "error"
            
            if amd_gpus and not self.gpu_info["rocm_available"]:
                issues.append("GPU AMD détecté mais ROCm non disponible")
                recommendations.append("Installez ROCm pour supporter les GPU AMD")
                severity = "warning"
        
        # Vérification 3: Mémoire GPU
        for gpu in self.gpu_info["gpus"]:
            if gpu["memory_total_mb"] < 4000:  # Moins de 4GB
                issues.append(f"GPU {gpu['name']} a peu de mémoire ({gpu['memory_total_mb']}MB)")
                recommendations.append("Utilisez des modèles quantifiés (Q4, Q5) pour économiser la mémoire GPU")
                severity = max(severity, "warning")
            
            if gpu["memory_free_mb"] < 2000:  # Moins de 2GB libre
                issues.append(f"GPU {gpu['name']} a peu de mémoire libre ({gpu['memory_free_mb']}MB)")
                recommendations.append("Fermez les applications utilisant le GPU ou redémarrez")
                severity = max(severity, "warning")
        
        # Vérification 4: Configuration Ollama
        ollama_config = self._check_ollama_gpu_config()
        if not ollama_config["gpu_enabled"]:
            issues.append("Ollama ne semble pas configuré pour utiliser le GPU")
            recommendations.append("Vérifiez la configuration Ollama avec les variables d'environnement")
            severity = "error"
        
        return {
            "system_info": self.system_info,
            "gpu_info": self.gpu_info,
            "issues": issues,
            "recommendations": recommendations,
            "severity": severity,
            "ollama_config": ollama_config
        }
    
    def _check_ollama_gpu_config(self) -> Dict:
        """Vérifie la configuration GPU d'Ollama."""
        
        config = {
            "gpu_enabled": False,
            "gpu_layers": 0,
            "environment_vars": {},
            "model_offload_status": {}
        }
        
        try:
            # Vérifier les variables d'environnement Ollama
            ollama_env_vars = [
                "OLLAMA_NUM_GPU",
                "OLLAMA_GPU_LAYERS", 
                "CUDA_VISIBLE_DEVICES",
                "OLLAMA_HOST",
                "OLLAMA_ORIGINS"
            ]
            
            for var in ollama_env_vars:
                value = os.environ.get(var)
                if value:
                    config["environment_vars"][var] = value
            
            # Tester une requête pour voir l'utilisation GPU
            test_payload = {
                "model": "mistral:7b-instruct",
                "messages": [{"role": "user", "content": "Test GPU"}],
                "options": {
                    "num_gpu": 50,  # Forcer l'utilisation GPU
                    "num_predict": 10
                },
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate", 
                json=test_payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                # Vérifier l'utilisation GPU pendant la génération
                initial_gpu_usage = self._get_current_gpu_usage()
                
                # Faire une vraie requête
                real_payload = {
                    "model": "mistral:7b-instruct", 
                    "prompt": "Expliquez l'intelligence artificielle en 50 mots.",
                    "options": {
                        "num_gpu": 50,
                        "num_predict": 60,
                        "temperature": 0.7
                    },
                    "stream": False
                }
                
                start_time = time.time()
                response = requests.post(f"{self.ollama_url}/api/generate", json=real_payload, timeout=60)
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    # Vérifier l'augmentation d'utilisation GPU
                    peak_gpu_usage = self._get_current_gpu_usage()
                    
                    gpu_increase = any(
                        peak_gpu_usage[i]["utilization"] > initial_gpu_usage[i]["utilization"] + 5
                        for i in range(min(len(peak_gpu_usage), len(initial_gpu_usage)))
                    )
                    
                    config["gpu_enabled"] = gpu_increase
                    config["processing_time"] = processing_time
                    config["initial_gpu_usage"] = initial_gpu_usage
                    config["peak_gpu_usage"] = peak_gpu_usage
                    
                    # Estimation des couches GPU utilisées
                    if gpu_increase:
                        avg_gpu_usage = sum(gpu["utilization"] for gpu in peak_gpu_usage) / len(peak_gpu_usage)
                        config["gpu_layers"] = int(avg_gpu_usage / 2)  # Estimation approximative
        
        except Exception as e:
            config["error"] = str(e)
        
        return config
    
    def _get_current_gpu_usage(self) -> List[Dict]:
        """Obtient l'utilisation actuelle des GPUs."""
        try:
            gpus = GPUtil.getGPUs()
            return [{
                "id": gpu.id,
                "utilization": gpu.load * 100,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "temperature": gpu.temperature
            } for gpu in gpus]
        except:
            return []
    
    def apply_gpu_optimizations(self) -> Dict:
        """Applique les optimisations GPU recommandées."""
        
        optimizations_applied = []
        errors = []
        
        # 1. Configuration des variables d'environnement
        try:
            gpu_count = self.gpu_info["gpu_count"]
            
            if gpu_count > 0:
                # Optimisations pour NVIDIA/CUDA
                if self.gpu_info["cuda_available"]:
                    os.environ["OLLAMA_NUM_GPU"] = str(gpu_count)
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
                    optimizations_applied.append(f"Configuré CUDA pour {gpu_count} GPU(s)")
                
                # Optimisations pour AMD/ROCm
                elif self.gpu_info["rocm_available"]:
                    os.environ["OLLAMA_NUM_GPU"] = str(gpu_count)
                    os.environ["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
                    optimizations_applied.append(f"Configuré ROCm pour {gpu_count} GPU(s)")
                
                # Optimisations pour Metal (macOS)
                elif self.gpu_info["metal_available"]:
                    os.environ["OLLAMA_NUM_GPU"] = "1"  # Metal utilise généralement 1 GPU intégré
                    optimizations_applied.append("Configuré Metal pour macOS")
        
        except Exception as e:
            errors.append(f"Erreur configuration variables d'environnement: {e}")
        
        # 2. Redémarrage d'Ollama pour appliquer les changements
        try:
            restart_result = self._restart_ollama_service()
            if restart_result["success"]:
                optimizations_applied.append("Service Ollama redémarré avec succès")
            else:
                errors.append(f"Échec redémarrage Ollama: {restart_result['error']}")
        
        except Exception as e:
            errors.append(f"Erreur redémarrage Ollama: {e}")
        
        # 3. Test de validation
        try:
            time.sleep(5)  # Attendre que le service redémarre
            validation = self._validate_gpu_setup()
            if validation["gpu_working"]:
                optimizations_applied.append("Validation GPU réussie")
            else:
                errors.append("Validation GPU échouée après optimisation")
        
        except Exception as e:
            errors.append(f"Erreur validation: {e}")
        
        return {
            "success": len(errors) == 0,
            "optimizations_applied": optimizations_applied,
            "errors": errors,
            "environment_vars": dict(os.environ),
            "recommendations": self._get_post_optimization_recommendations()
        }
    
    def _restart_ollama_service(self) -> Dict:
        """Redémarre le service Ollama."""
        
        try:
            # Arrêt du service
            if platform.system() == "Windows":
                stop_cmd = ["taskkill", "/F", "/IM", "ollama.exe"]
                start_cmd = ["ollama", "serve"]
            else:
                stop_cmd = ["pkill", "-f", "ollama"]
                start_cmd = ["ollama", "serve"]
            
            # Arrêter Ollama
            try:
                subprocess.run(stop_cmd, capture_output=True, timeout=10)
                time.sleep(2)
            except:
                pass  # Pas grave si déjà arrêté
            
            # Redémarrer Ollama en arrière-plan
            if platform.system() == "Windows":
                process = subprocess.Popen(start_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                process = subprocess.Popen(start_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Attendre quelques secondes et vérifier si le service répond
            time.sleep(5)
            
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    return {"success": True, "pid": process.pid}
                else:
                    return {"success": False, "error": f"Service non réactif (HTTP {response.status_code})"}
            except:
                return {"success": False, "error": "Service non accessible après redémarrage"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _validate_gpu_setup(self) -> Dict:
        """Valide que la configuration GPU fonctionne."""
        
        validation = {
            "gpu_working": False,
            "performance_improvement": False,
            "baseline_time": 0,
            "gpu_time": 0,
            "gpu_usage_detected": False
        }
        
        try:
            # Test de base (CPU)
            cpu_payload = {
                "model": "mistral:7b-instruct",
                "prompt": "Expliquez brièvement l'IA.",
                "options": {
                    "num_gpu": 0,  # Force CPU
                    "num_predict": 30
                },
                "stream": False
            }
            
            start_time = time.time()
            response = requests.post(f"{self.ollama_url}/api/generate", json=cpu_payload, timeout=60)
            validation["baseline_time"] = time.time() - start_time
            
            if response.status_code != 200:
                return validation
            
            # Test GPU
            gpu_payload = {
                "model": "mistral:7b-instruct",
                "prompt": "Expliquez brièvement l'IA.",
                "options": {
                    "num_gpu": 50,  # Force GPU
                    "num_predict": 30
                },
                "stream": False
            }
            
            initial_gpu_state = self._get_current_gpu_usage()
            
            start_time = time.time()
            response = requests.post(f"{self.ollama_url}/api/generate", json=gpu_payload, timeout=60)
            validation["gpu_time"] = time.time() - start_time
            
            peak_gpu_state = self._get_current_gpu_usage()
            
            if response.status_code == 200:
                # Vérifier l'augmentation d'utilisation GPU
                gpu_usage_increase = any(
                    peak_gpu_state[i]["utilization"] > initial_gpu_state[i]["utilization"] + 10
                    for i in range(min(len(peak_gpu_state), len(initial_gpu_state)))
                ) if initial_gpu_state and peak_gpu_state else False
                
                validation["gpu_usage_detected"] = gpu_usage_increase
                validation["gpu_working"] = gpu_usage_increase
                validation["performance_improvement"] = validation["gpu_time"] < validation["baseline_time"] * 0.8
        
        except Exception as e:
            validation["error"] = str(e)
        
        return validation
    
    def _get_post_optimization_recommendations(self) -> List[str]:
        """Fournit des recommandations post-optimisation."""
        
        recommendations = []
        
        # Recommandations basées sur le matériel
        total_vram = sum(gpu["memory_total_mb"] for gpu in self.gpu_info["gpus"])
        
        if total_vram < 6000:  # Moins de 6GB VRAM total
            recommendations.extend([
                "🎯 Utilisez des modèles quantifiés (mistral:7b-instruct-q4_0) pour économiser la VRAM",
                "⚡ Évitez les gros modèles comme llama3:70b sur votre configuration",
                "🔧 Considérez l'augmentation de la mémoire GPU si possible"
            ])
        
        elif total_vram < 12000:  # 6-12GB VRAM
            recommendations.extend([
                "✅ Configuration adaptée aux modèles 7B et 13B",
                "🎯 Testez les modèles Mixtral:8x7B pour de meilleures performances",
                "⚖️ Équilibrez entre qualité et vitesse selon vos besoins"
            ])
        
        else:  # Plus de 12GB VRAM
            recommendations.extend([
                "🚀 Configuration excellente pour tous les modèles disponibles",
                "🎯 Testez les modèles 70B pour des analyses très poussées",
                "🔀 Expérimentez avec plusieurs modèles en parallèle"
            ])
        
        # Recommandations de configuration
        recommendations.extend([
            "📊 Surveillez l'utilisation GPU avec 'nvidia-smi' ou 'rocm-smi'",
            "🔄 Redémarrez Ollama après changement de modèles lourds",
            "⚙️ Ajustez 'num_gpu' selon la taille du document à traiter",
            "🌡️ Surveillez la température GPU lors de traitements intensifs"
        ])
        
        return recommendations
    
    def generate_optimal_config(self, model_name: str, document_size: str = "medium") -> Dict:
        """Génère une configuration optimale pour un modèle donné."""
        
        config = {
            "model": model_name,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_thread": min(8, self.system_info["cpu_count"])
            },
            "timeout": 85,
            "use_chunking": False
        }
        
        # Configuration GPU basée sur le matériel disponible
        if self.gpu_info["gpus_available"]:
            total_vram = sum(gpu["memory_total_mb"] for gpu in self.gpu_info["gpus"])
            
            # Estimation des couches GPU selon la VRAM
            if "7b" in model_name.lower():
                if total_vram >= 8000:
                    config["options"]["num_gpu"] = 50
                elif total_vram >= 4000:
                    config["options"]["num_gpu"] = 35
                else:
                    config["options"]["num_gpu"] = 20
            
            elif "13b" in model_name.lower():
                if total_vram >= 16000:
                    config["options"]["num_gpu"] = 50
                elif total_vram >= 8000:
                    config["options"]["num_gpu"] = 30
                else:
                    config["options"]["num_gpu"] = 15
            
            elif "70b" in model_name.lower():
                if total_vram >= 48000:
                    config["options"]["num_gpu"] = 50
                elif total_vram >= 24000:
                    config["options"]["num_gpu"] = 25
                else:
                    config["options"]["num_gpu"] = 0  # CPU uniquement
            
            else:  # Modèles non standard
                config["options"]["num_gpu"] = min(30, int(total_vram / 200))
        
        else:
            config["options"]["num_gpu"] = 0  # CPU uniquement
        
        # Ajustements selon la taille du document
        size_configs = {
            "small": {
                "num_predict": 500,
                "num_ctx": 2048,
                "timeout": 30
            },
            "medium": {
                "num_predict": 1500,
                "num_ctx": 4096,
                "timeout": 60
            },
            "large": {
                "num_predict": 3000,
                "num_ctx": 8192,
                "timeout": 85,
                "use_chunking": True
            },
            "xlarge": {
                "num_predict": 2000,  # Réduit pour éviter timeout
                "num_ctx": 4096,
                "timeout": 85,
                "use_chunking": True
            }
        }
        
        size_config = size_configs.get(document_size, size_configs["medium"])
        config["options"].update({k: v for k, v in size_config.items() if k != "use_chunking"})
        config["use_chunking"] = size_config.get("use_chunking", False)
        
        # Optimisation batch selon le GPU
        if config["options"]["num_gpu"] > 0:
            config["options"]["num_batch"] = 512
        else:
            config["options"]["num_batch"] = 256
        
        return config
    
    def monitor_gpu_performance(self, duration: int = 60) -> Dict:
        """Surveille les performances GPU pendant une durée donnée."""
        
        monitoring_data = {
            "duration": duration,
            "samples": [],
            "start_time": time.time(),
            "end_time": None
        }
        
        print(f"🔍 Surveillance GPU pendant {duration} secondes...")
        
        try:
            start_time = time.time()
            
            while time.time() - start_time < duration:
                sample = {
                    "timestamp": time.time() - start_time,
                    "gpus": self._get_current_gpu_usage(),
                    "system_ram": psutil.virtual_memory().percent,
                    "cpu_usage": psutil.cpu_percent()
                }
                
                monitoring_data["samples"].append(sample)
                time.sleep(2)  # Échantillonnage toutes les 2 secondes
            
            monitoring_data["end_time"] = time.time()
            
            # Analyse des données
            analysis = self._analyze_monitoring_data(monitoring_data)
            monitoring_data["analysis"] = analysis
        
        except KeyboardInterrupt:
            print("🛑 Surveillance interrompue par l'utilisateur")
            monitoring_data["end_time"] = time.time()
        
        except Exception as e:
            monitoring_data["error"] = str(e)
        
        return monitoring_data
    
    def _analyze_monitoring_data(self, data: Dict) -> Dict:
        """Analyse les données de surveillance."""
        
        if not data["samples"]:
            return {"error": "Aucune donnée à analyser"}
        
        analysis = {
            "gpu_stats": {},
            "system_stats": {},
            "recommendations": []
        }
        
        # Analyse GPU
        for gpu_id in range(self.gpu_info["gpu_count"]):
            gpu_utilizations = [
                sample["gpus"][gpu_id]["utilization"] 
                for sample in data["samples"] 
                if len(sample["gpus"]) > gpu_id
            ]
            
            if gpu_utilizations:
                analysis["gpu_stats"][f"gpu_{gpu_id}"] = {
                    "avg_utilization": sum(gpu_utilizations) / len(gpu_utilizations),
                    "max_utilization": max(gpu_utilizations),
                    "min_utilization": min(gpu_utilizations),
                    "usage_variance": max(gpu_utilizations) - min(gpu_utilizations)
                }
        
        # Analyse système
        ram_usages = [sample["system_ram"] for sample in data["samples"]]
        cpu_usages = [sample["cpu_usage"] for sample in data["samples"]]
        
        analysis["system_stats"] = {
            "avg_ram_usage": sum(ram_usages) / len(ram_usages),
            "max_ram_usage": max(ram_usages),
            "avg_cpu_usage": sum(cpu_usages) / len(cpu_usages),
            "max_cpu_usage": max(cpu_usages)
        }
        
        # Recommandations basées sur l'analyse
        for gpu_id, stats in analysis["gpu_stats"].items():
            if stats["avg_utilization"] < 10:
                analysis["recommendations"].append(
                    f"⚠️ {gpu_id}: Utilisation faible ({stats['avg_utilization']:.1f}%) - Vérifiez la configuration"
                )
            elif stats["avg_utilization"] > 90:
                analysis["recommendations"].append(
                    f"🔥 {gpu_id}: Utilisation élevée ({stats['avg_utilization']:.1f}%) - Surveillez la température"
                )
            elif stats["usage_variance"] > 50:
                analysis["recommendations"].append(
                    f"📊 {gpu_id}: Utilisation variable - Normal lors de traitements par lots"
                )
        
        if analysis["system_stats"]["max_ram_usage"] > 85:
            analysis["recommendations"].append("💾 RAM système élevée - Fermez les applications inutiles")
        
        if analysis["system_stats"]["avg_cpu_usage"] > 80:
            analysis["recommendations"].append("🔄 CPU fortement sollicité - Réduisez num_thread")
        
        return analysis

# =============================================================================
# FONCTIONS D'INTÉGRATION AVEC L'INTERFACE
# =============================================================================

def create_gpu_configurator(ollama_url: str = "http://localhost:11434"):
    """Factory function pour créer un configurateur GPU."""
    return OllamaGPUConfigurator(ollama_url)

def quick_gpu_diagnostic() -> Dict:
    """Diagnostic rapide des problèmes GPU."""
    configurator = OllamaGPUConfigurator()
    return configurator.diagnose_gpu_issues()

def auto_optimize_gpu_config(ollama_url: str = "http://localhost:11434") -> Dict:
    """Optimisation automatique de la configuration GPU."""
    configurator = OllamaGPUConfigurator(ollama_url)
    return configurator.apply_gpu_optimizations()

def get_optimal_config_for_model(model_name: str, document_size: str = "medium") -> Dict:
    """Obtient la configuration optimale pour un modèle."""
    configurator = OllamaGPUConfigurator()
    return configurator.generate_optimal_config(model_name, document_size)

def start_gpu_monitoring(duration: int = 60) -> Dict:
    """Démarre la surveillance GPU."""
    configurator = OllamaGPUConfigurator()
    return configurator.monitor_gpu_performance(duration)

# =============================================================================
# INTERFACE GRADIO POUR DIAGNOSTIC GPU
# =============================================================================

def create_gpu_diagnostic_interface():
    """Crée une interface Gradio pour le diagnostic GPU."""
    
    def run_diagnostic():
        """Lance le diagnostic GPU."""
        result = quick_gpu_diagnostic()
        
        # Formatage pour l'affichage
        report = f"""
🖥️ **INFORMATIONS SYSTÈME**
- OS: {result['system_info']['os']} {result['system_info']['architecture']}
- CPU: {result['system_info']['cpu_count']} cœurs
- RAM: {result['system_info']['total_ram_gb']} GB total, {result['system_info']['available_ram_gb']} GB disponible

🎮 **INFORMATIONS GPU**
- GPUs détectés: {result['gpu_info']['gpu_count']}
- CUDA disponible: {'✅' if result['gpu_info']['cuda_available'] else '❌'}
- ROCm disponible: {'✅' if result['gpu_info']['rocm_available'] else '❌'}
- Metal disponible: {'✅' if result['gpu_info']['metal_available'] else '❌'}
"""
        
        if result['gpu_info']['gpus']:
            report += "\n**DÉTAILS DES GPUS:**\n"
            for gpu in result['gpu_info']['gpus']:
                report += f"- {gpu['name']}: {gpu['memory_total_mb']}MB, Utilisation: {gpu['utilization']:.1f}%\n"
        
        if result['issues']:
            report += f"\n⚠️ **PROBLÈMES DÉTECTÉS ({result['severity'].upper()}):**\n"
            for issue in result['issues']:
                report += f"- {issue}\n"
        
        if result['recommendations']:
            report += "\n💡 **RECOMMANDATIONS:**\n"
            for rec in result['recommendations']:
                report += f"- {rec}\n"
        
        return report
    
    def apply_optimizations():
        """Applique les optimisations GPU."""
        result = auto_optimize_gpu_config()
        
        report = f"**STATUT:** {'✅ SUCCÈS' if result['success'] else '❌ ÉCHEC'}\n\n"
        
        if result['optimizations_applied']:
            report += "**OPTIMISATIONS APPLIQUÉES:**\n"
            for opt in result['optimizations_applied']:
                report += f"- {opt}\n"
        
        if result['errors']:
            report += "\n**ERREURS:**\n"
            for error in result['errors']:
                report += f"- {error}\n"
        
        if result['recommendations']:
            report += "\n**RECOMMANDATIONS POST-OPTIMISATION:**\n"
            for rec in result['recommendations']:
                report += f"- {rec}\n"
        
        return report
    
    return run_diagnostic, apply_optimizations

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

def example_gpu_configuration():
    """Exemple d'utilisation du configurateur GPU."""
    
    print("🚀 Diagnostic GPU Ollama")
    
    # Diagnostic initial
    configurator = OllamaGPUConfigurator()
    diagnosis = configurator.diagnose_gpu_issues()
    
    print(f"\n📊 Système: {diagnosis['system_info']['os']} - {diagnosis['gpu_info']['gpu_count']} GPU(s)")
    print(f"🎯 Sévérité: {diagnosis['severity'].upper()}")
    
    if diagnosis['issues']:
        print("\n⚠️ Problèmes détectés:")
        for issue in diagnosis['issues']:
            print(f"  - {issue}")
    
    # Application des optimisations
    if diagnosis['severity'] in ['warning', 'error']:
        print("\n🔧 Application des optimisations...")
        optimization_result = configurator.apply_gpu_optimizations()
        
        if optimization_result['success']:
            print("✅ Optimisations appliquées avec succès")
        else:
            print("❌ Échec des optimisations:")
            for error in optimization_result['errors']:
                print(f"  - {error}")
    
    # Configuration optimale pour un modèle
    optimal_config = configurator.generate_optimal_config("mistral:7b-instruct", "medium")
    print(f"\n⚙️ Configuration optimale pour Mistral 7B:")
    print(f"  - GPU Layers: {optimal_config['options']['num_gpu']}")
    print(f"  - Contexte: {optimal_config['options']['num_ctx']}")
    print(f"  - Chunking: {'Activé' if optimal_config['use_chunking'] else 'Désactivé'}")

if __name__ == "__main__":
    example_gpu_configuration()