#!/bin/bash
set -e

# 1. Mettre à jour le système
apt-get update && apt-get upgrade -y

# 2. Installer curl et OpenSSH
apt-get install -y curl openssh-server

# 3. Installer Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 4. Créer un utilisateur (si nécessaire)
useradd -m -s /bin/bash runpod
echo "runpod:changeme" | chpasswd

# 5. Activer SSH
mkdir -p /var/run/sshd
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
service ssh start

# 6. Vérifier Ollama
ollama --version

echo "✅ Installation terminée. SSH actif sur port 22."

