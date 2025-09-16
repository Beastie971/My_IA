
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

# Paquets de base
RUN apt-get update && apt-get install -y --no-install-recommends     ca-certificates curl openssh-server sudo gnupg lsb-release     python3 python3-pip python3-venv  && rm -rf /var/lib/apt/lists/*

# Installer Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Configurer SSH (clé uniquement)
RUN mkdir -p /var/run/sshd /root/.ssh  && chmod 700 /root/.ssh  && sed -i 's/#\?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config  && sed -i 's/#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config  && sed -i 's/#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# PyTorch CUDA 12.4 (wheels officiels)
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
RUN python3 -m pip install --upgrade pip  && pip3 install --no-cache-dir torch torchvision torchaudio --index-url ${TORCH_INDEX_URL}

# Ollama écoute partout
ENV OLLAMA_HOST=0.0.0.0:11434

# Entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22 11434
VOLUME ["/root/.ollama"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
