
# Ollama + SSH + PyTorch CUDA (Runpod-ready) — Entrypoint bash forcé

Cette variante force l'exécution de l'entrypoint avec **/bin/bash** afin d'utiliser `set -euo pipefail` et d'éviter l'erreur `set: Illegal option -o pipefail` sous `/bin/sh` (dash).

## Build
```bash
docker build --platform linux/amd64 -t <user>/<repo>:ollama-ssh-torch-cu124 -f Dockerfile .
```

## Run local (GPU NVIDIA)
```bash
docker run -d --name ollama-gpu --gpus all   -p 2222:22 -p 11434:11434 <user>/<repo>:ollama-ssh-torch-cu124
```

## Déploiement Runpod
- Image : `<user>/<repo>:ollama-ssh-torch-cu124`
- Expose **TCP Ports** : `22,11434`
- ENV : `SSH_PUBLIC_KEY` (optionnel), `OLLAMA_HOST=0.0.0.0:11434` (déjà défini)
- Volume : `/root/.ollama`

## Notes
- PyTorch CUDA 12.4 via `--index-url https://download.pytorch.org/whl/cu124`.
- L'entrée `ENTRYPOINT ["/bin/bash", "/usr/local/bin/entrypoint.sh"]` garantit l'exécution sous bash.
