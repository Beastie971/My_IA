export IMAGE_LOCAL=ollama-cublas:0.1

# (Optionnel) voir le Dockerfile réellement utilisé et toutes les étapes
docker build --pull --no-cache --progress=plain -t "$IMAGE_LOCAL" -f Dockerfile .
# Si tu as plusieurs Dockerfile, précise bien -f et le bon chemin.

docker tag ${IMAGE_LOCAL}  beastie971/$IMAGE_LOCAL
