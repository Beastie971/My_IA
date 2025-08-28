#!/usr/bin/env python3

import sys, os, pickle, json
from typing import Any, Iterable

POSSIBLE_TEXT_KEYS = {
    "text", "content", "raw_text", "document", "full_text",
    "texte", "contenu", "pages_text", "ocr_text"
}

def iter_strings(obj: Any) -> Iterable[str]:
    """Récupère récursivement toutes les chaînes trouvées dans un objet Python."""
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        # 1) clés usuelles susceptibles de contenir le texte intégral
        for k in list(obj.keys()):
            if isinstance(k, str) and k.lower() in POSSIBLE_TEXT_KEYS:
                v = obj[k]
                if isinstance(v, str):
                    yield v
                elif isinstance(v, (list, tuple)):
                    for s in v:
                        if isinstance(s, str):
                            yield s
        # 2) fallback: on parcourt toutes les valeurs
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from iter_strings(v)

def load_text_from_pickle(path: str) -> str:
    with open(path, "rb") as f:
        data = pickle.load(f)
    # 1) Si c'est déjà une chaîne
    if isinstance(data, str):
        return data
    # 2) Si c'est un dict avec une clé connue
    if isinstance(data, dict):
        for key in POSSIBLE_TEXT_KEYS:
            for k in data.keys():
                if isinstance(k, str) and k.lower() == key:
                    v = data[k]
                    if isinstance(v, str):
                        return v
                    if isinstance(v, (list, tuple)):
                        parts = [s for s in v if isinstance(s, str)]
                        if parts:
                            return "\n".join(parts)
        # 3) Heuristique: concaténer les strings trouvées (limité pour éviter bruit)
        strings = list(iter_strings(data))
        # Trier grossièrement par longueur décroissante et prendre les plus longues
        strings.sort(key=len, reverse=True)
        if strings:
            # On prend les 3 plus longues pour éviter le bruit
            return "\n\n".join(strings[:3])
    # 4) Si c'est une liste de strings
    if isinstance(data, (list, tuple)) and all(isinstance(x, str) for x in data):
        return "\n".join(data)
    raise ValueError("Impossible d'extraire du texte exploitable depuis le pickle.")

def main():
    if len(sys.argv) != 2:
        print("Usage: pkl_to_text.py <fichier.pkl>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Fichier introuvable: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        txt = load_text_from_pickle(path)
        # normalisation basique
        txt = txt.replace("\r\n", "\n").replace("\r", "\n").strip()
        print(txt)
    except Exception as e:
        print(f"Erreur extraction pickle: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

