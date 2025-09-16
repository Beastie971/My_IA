
# Ollama + SSH + PyTorch CUDA (Runpod-ready)

Deux variantes :

- `Dockerfile` : Ubuntu 22.04 minimal, PyTorch CUDA 12.4 via wheels, pas de toolkit CUDA dans l'image.
- `Dockerfile.devel` : Base `nvidia/cuda:12.4.1-devel-ubuntu22.04` avec `nvcc`, utile pour compiler des ops custom.

## Build local

```bash
docker build -t ollama-ssh-torch:cu124 -f Dockerfile .
# ou la variante devel
# docker build -t ollama-ssh-torch:cu124-devel -f Dockerfile.devel .
```

## Run local (GPU NVIDIA)

```bash
docker run -d --name ollama-gpu --gpus all -p 2222:22 -p 11434:11434 ollama-ssh-torch:cu124
```

Vérifier PyTorch GPU :

```bash
docker exec -it ollama-gpu python3 - <<'PY'
import torch
print('CUDA dispo:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
```

Vérifier l'API Ollama :

```bash
curl http://127.0.0.1:11434/api/tags
```

## Déploiement Runpod

1. Pousser l'image vers votre registre :

```bash
docker tag ollama-ssh-torch:cu124 <user>/<repo>:ollama-ssh-torch-cu124
docker push <user>/<repo>:ollama-ssh-torch-cu124
```

2. Créer un **Template Runpod** pointant sur l'image.

- Expose **TCP Ports** : `22,11434`
- Variables d'env (optionnel) :
  - `SSH_PUBLIC_KEY` : votre clé publique pour activer SSH (PasswordAuthentication désactivé)
  - `OLLAMA_HOST=0.0.0.0:11434` (déjà défini dans l'image)
- Montez un Network Volume sur `/root/.ollama` pour persister les modèles.

3. Déployer un **Pod GPU** et se connecter :

- SSH (proxy) : via l'onglet **Connect → SSH** de la console Runpod
- API : via proxy `https://<PODID>-11434.proxy.runpod.net` ou via IP/port public si activé

## Notes

- Les wheels PyTorch CUDA 12.4 sont fournis via l'index `https://download.pytorch.org/whl/cu124`.
- Sur Runpod, le runtime GPU (NVIDIA Container Toolkit) est déjà configuré pour les Pods GPU.
- L'API Ollama écoute par défaut sur `11434` et nous la lions à `0.0.0.0` via `OLLAMA_HOST`.

## Sécurité

- SSH est limité à l'authentification par **clé**. Ajoutez `SSH_PUBLIC_KEY` côté Template ou via votre profil Runpod.
- Pensez à restreindre l'accès public à l'API (reverse proxy, auth) si nécessaire.
