#!/usr/bin/env bash
set -euo pipefail

# Injecter une clé publique si fournie via variable d'env (pratique sur Runpod)
if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
  mkdir -p /root/.ssh
  echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi

# Générer les clés hôte SSH si absentes
ssh-keygen -A

# Démarrer sshd en arrière-plan
/usr/sbin/sshd

# Lancer Ollama en avant-plan (garde le conteneur vivant)
exec ollama serve

