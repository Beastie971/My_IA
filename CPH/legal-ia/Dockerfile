# Dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Outils de base
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Installer Ollama (binaire officiel)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Forcer l'utilisation du backend cuBLAS (GPU)
ENV OLLAMA_LLM_LIBRARY=cublas
ENV OLLAMA_HOST=0.0.0.0

EXPOSE 11434
VOLUME ["/root/.ollama"]

# Healthcheck basique (ollama prêt quand /api/tags répond)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=5 \
CMD curl -fsS http://127.0.0.1:11434/api/tags >/dev/null || exit 1

CMD ["ollama", "serve"]

