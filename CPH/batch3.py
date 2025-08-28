#!/usr/bin/env python3

import argparse, json, pickle, os, sys
from urllib.request import Request, urlopen

# Réutilise la même heuristique d'extraction
from pkl2txt import load_text_from_pickle  # place les deux scripts dans le même dossier

def main():
    ap = argparse.ArgumentParser(description="Prud'hommes via Ollama /api/chat (entrée .pkl)")
    ap.add_argument("--model", default="prudhommes-8b")
    ap.add_argument("--file", required=True, help="Fichier pickle contenant le texte")
    ap.add_argument("--output", required=True, help="Chemin du fichier de sortie")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repeat_penalty", type=float, default=1.1)
    ap.add_argument("--num_ctx", type=int, default=8192)
    ap.add_argument("--num_predict", type=int, default=2048)
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"Fichier introuvable: {args.file}", file=sys.stderr)
        sys.exit(1)

    user_msg = load_text_from_pickle(args.file).replace("\r\n", "\n").strip()

    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": user_msg}
        ],
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repeat_penalty": args.repeat_penalty,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict
        },
        "stream": False
    }

    req = Request(f"{args.host}/api/chat",
                  data=json.dumps(payload).encode("utf-8"),
                  headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        data = json.load(resp)
    text = data.get("message", {}).get("content", "")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Résultat enregistré dans : {args.output}")

if __name__ == "__main__":
    main()

