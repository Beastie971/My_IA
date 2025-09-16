
#!/usr/bin/env bash
set -euo pipefail

# Injecte une clé publique SSH si fournie via variable d'env
if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
  mkdir -p /root/.ssh
  echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi

# Génère les clés hôte SSH si absentes
ssh-keygen -A

# Démarre sshd en arrière-plan
/usr/sbin/sshd

# Lance Ollama en avant-plan
exec ollama serve
