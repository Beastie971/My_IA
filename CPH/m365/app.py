#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, re, json, time, math, hashlib, shutil, zipfile
import typing as t
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gradio as gr
import requests

# Pré-traitement documents
from PIL import Image
import fitz  # PyMuPDF
from docx import Document  # python-docx

# OCR optionnel (activé si dispo + binaire tesseract installé)
with_ocr = False
try:
    import pytesseract
    with_ocr = True
except Exception:
    with_ocr = False

# ============
# CONFIG
# ============
DEFAULT_OLLAMA_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "https://tszwlwj46df2kj-11434.proxy.runpod.net"
).rstrip("/")

CACHE_DIR = Path("cache_gradio_ocr_analysis")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Seuils d'aiguillage
DEFAULT_MAX_DIRECT_CHARS = 10_000            # < 10k -> direct
DEFAULT_CHUNK_MAX_CHARS = 4_500              # chunk size ~ 4.5k
DEFAULT_CHUNK_OVERLAP = 400                  # overlap ~ 400
DEFAULT_NUM_CTX = 8192                       # contexte par défaut ollama
DEFAULT_TEMPERATURE = 0.2

# ============
# UTILITAIRES
# ============
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

# ============
# OLLAMA API
# ============
def list_models(base_url: str) -> t.Tuple[t.List[str], str]:
    """
    Retourne (liste_de_noms, message_statut).
    """
    url = base_url.rstrip("/") + "/api/tags"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        models = sorted(models)
        msg = f"✅ {len(models)} modèle(s) trouvé(s) sur {base_url}"
        return models, msg
    except Exception as e:
        return [], f"❌ Échec /api/tags sur {base_url} : {e}"

def ollama_generate_stream(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_gpu: int = 999,
    top_p: float = 0.9,
    max_tokens: int = 1024,
) -> t.Iterator[str]:
    """
    Stream de génération (ligne JSON par chunk).
    Utilise /api/generate de l'API Ollama.
    """
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "top_p": top_p,
            # Forcer offload GPU côté serveur si dispo (ignoré si non supporté)
            "num_gpu": num_gpu,
            "gpu_layers": num_gpu,
        }
    }
    with requests.post(url, json=payload, timeout=600, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
                if "response" in obj and obj["response"]:
                    yield obj["response"]
                if obj.get("done"):
                    break
            except Exception:
                # si ligne JSON imparfaite, on ignore
                continue

# ============
# EXTRACTION TEXTE & OCR
# ============
def extract_text_from_pdf(pdf_path: Path, force_ocr: bool = False, ocr_lang: str = "fra") -> str:
    """
    Extraction texte PDF via PyMuPDF. Si force_ocr ou peu de texte -> OCR page par page (si dispo).
    """
    doc = fitz.open(pdf_path)
    texts = []
    # première passe : texte natif
    native_text_len = 0
    for page in doc:
        t = page.get_text("text")
        texts.append(t)
        native_text_len += len(t)

    text_all = "\n".join(texts).strip()

    if (not force_ocr) and native_text_len > 1000:
        return text_all
    # OCR (si dispo)
    if not with_ocr:
        # On avertit : OCR indisponible
        return text_all if text_all else "[OCR indisponible] Installez tesseract-ocr + pytesseract pour OCR."

    # Rasterize + OCR
    ocr_texts = []
    for page in doc:
        pix = page.get_pixmap(dpi=300, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            txt = pytesseract.image_to_string(img, lang=ocr_lang)
        except Exception as e:
            txt = f"[Erreur OCR: {e}]"
        ocr_texts.append(txt)
    return "\n".join(ocr_texts).strip()

def extract_text_from_docx(docx_path: Path) -> str:
    doc = Document(str(docx_path))
    paras = []
    for p in doc.paragraphs:
        paras.append(p.text)
    return "\n".join(paras).strip()

def extract_text_from_image(img_path: Path, ocr_lang: str = "fra") -> str:
    # OCR image (si dispo), sinon message
    if with_ocr:
        img = Image.open(img_path)
        return pytesseract.image_to_string(img, lang=ocr_lang).strip()
    else:
        return "[OCR indisponible] Installez tesseract-ocr + pytesseract pour OCR."

def extract_text(file_path: Path, force_ocr: bool, ocr_lang: str) -> str:
    suf = file_path.suffix.lower()
    if suf == ".pdf":
        return extract_text_from_pdf(file_path, force_ocr=force_ocr, ocr_lang=ocr_lang)
    elif suf in [".txt", ".log", ".md"]:
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return file_path.read_text(encoding="latin-1", errors="ignore")
    elif suf == ".docx":
        return extract_text_from_docx(file_path)
    elif suf in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        return extract_text_from_image(file_path, ocr_lang=ocr_lang)
    else:
        return f"[Format non géré] {file_path.name}"

# ============
# NETTOYAGE & ANONYMISATION
# ============
def normalize_whitespace(s: str) -> str:
    # collapse espaces multiples, normaliser \r\n -> \n
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # supprimer espaces fin de ligne
    s = re.sub(r"[ \t]+$", "", s, flags=re.MULTILINE)
    # lignes vides multiples
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def fix_hyphenation(s: str) -> str:
    # recompose mots coupés en fin de ligne: "installa-\ntion" -> "installation"
    return re.sub(r"(\w)-\n(\w)", r"\1\2", s)

def remove_repeated_headers_footers(s: str) -> str:
    # heuristique simple sur blocs de début/fin de page répétés
    pages = s.split("\f") if "\f" in s else s.split("\n\n\n")
    if len(pages) < 4:
        return s
    def top_lines(page):  return "\n".join(page.splitlines()[:3]).strip()
    def bottom_lines(page): return "\n".join(page.splitlines()[-3:]).strip()

    tops = [top_lines(p) for p in pages if p.strip()]
    bots = [bottom_lines(p) for p in pages if p.strip()]
    def frequent(blocks):
        freq = {}
        for b in blocks:
            if not b: continue
            freq[b] = freq.get(b, 0) + 1
        if not freq: return None
        b, n = max(freq.items(), key=lambda kv: kv[1])
        return b if n >= max(2, int(0.6 * len(pages))) else None

    top_rep = frequent(tops)
    bot_rep = frequent(bots)
    if not top_rep and not bot_rep:
        return s

    new_pages = []
    for p in pages:
        lines = p.splitlines()
        if top_rep and "\n".join(lines[:3]).strip() == top_rep.strip():
            lines = lines[3:]
        if bot_rep and "\n".join(lines[-3:]).strip() == bot_rep.strip():
            lines = lines[:-3]
        new_pages.append("\n".join(lines))
    return "\n\n".join(new_pages).strip()

def anonymize(s: str) -> str:
    # emails
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", s)
    # téléphones (FR approx)
    s = re.sub(r"(\+33|0)\s*1-9{4}", "[PHONE]", s)
    # IBAN (FR)
    s = re.sub(r"\bFR\d{2}[A-Z0-9]{11,27}\b", "[IBAN]", s, flags=re.IGNORECASE)
    # montants €
    s = re.sub(r"\b\d{1,3}(?:[ .]\d{3})*(?:,\d+)?\s?(?:€|EUR)\b", "[AMOUNT]", s, flags=re.IGNORECASE)
    # dates (FR simples)
    s = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]", s)
    # Noms propres en MAJ (heuristique) -> [NAME] si >= 2 mots en MAJ
    s = re.sub(r"\b([A-ZÉÈÀÂÙÛÖÏÇ]{2,}\s+[A-ZÉÈÀÂÙÛÖÏÇ]{2,})\b", "[NAME]", s)
    return s

def preprocess_text(raw: str, do_clean: bool, do_anonym: bool) -> str:
    text = raw
    if do_clean:
        text = normalize_whitespace(text)
        text = fix_hyphenation(text)
        text = remove_repeated_headers_footers(text)
    if do_anonym:
        text = anonymize(text)
    return text.strip()

# ============
# CHUNKING
# ============
def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_MAX_CHARS, overlap: int = DEFAULT_CHUNK_OVERLAP) -> t.List[str]:
    parts = []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        cut = text.rfind("\n\n", start, end)
        if cut == -1 or cut <= start + 800:
            cut = text.rfind(".", start, end)
            if cut == -1 or cut <= start:
                cut = end
        parts.append(text[start:cut].strip())
        start = max(cut - overlap, 0)
        if end == n:
            break
    # dedup final
    dedup = []
    for p in parts:
        if not dedup or p not in dedup[-1]:
            dedup.append(p)
    return [p for p in dedup if p]

# ============
# PROMPTS PAR DÉFAUT (éditables dans l'UI)
# ============
DEFAULT_DIRECT_PROMPT = """Rôle: Analyste. Tâche: Analyser le document ci-dessous et produire un exposé clair et structuré.
Exigences:
- Synthèse fidèle, sans inventer de faits
- Mettre en avant les points saillants, listes raisonnables
- Mentionner ambiguïtés, manques et pistes de vérification
Réponds en français.

=== DOCUMENT ===
{DOCUMENT}
"""

DEFAULT_CHUNK_PROMPT = """Rôle: Analyste. Tu reçois un extrait (chunk {I}/{N}) du document.
Tâche: Résume fidèlement CE CHUNK UNIQUEMENT, points clés, données factuelles, éléments juridiques/techniques s'il y en a.
Ne fais aucune spéculation hors du chunk.
Texte:
{CHUNK}
"""

DEFAULT_AGG_PROMPT = """Rôle: Analyste. Tu reçois des résumés par chunk (JSONL ou blocs).
Objectif: Produire une synthèse globale cohérente et non redondante, en structurant en sections (Contexte / Points clés / Ambiguïtés / Recos).
Réponds en français.

=== RÉSUMÉS CHUNKS ===
{SUMMARIES}
"""

# ============
# CACHE
# ============
def cache_key_for(
    file_hash: str,
    do_clean: bool,
    do_anonym: bool,
    force_ocr: bool,
    ocr_lang: str,
    mode: str,                   # "direct" | "chunks"
    model_direct: str,
    model_chunk: str,
    model_agg: str,
    direct_prompt: str,
    chunk_prompt: str,
    agg_prompt: str,
    chunk_max: int,
    chunk_overlap: int,
) -> str:
    payload = json.dumps({
        "file_hash": file_hash,
        "do_clean": do_clean,
        "do_anonym": do_anonym,
        "force_ocr": force_ocr,
        "ocr_lang": ocr_lang,
        "mode": mode,
        "model_direct": model_direct,
        "model_chunk": model_chunk,
        "model_agg": model_agg,
        "direct_prompt_sha": sha256_text(direct_prompt),
        "chunk_prompt_sha": sha256_text(chunk_prompt),
        "agg_prompt_sha": sha256_text(agg_prompt),
        "chunk_max": chunk_max,
        "chunk_overlap": chunk_overlap,
    }, sort_keys=True)
    return sha256_text(payload)

def cache_put(key: str, content: str):
    p = CACHE_DIR / f"{key}.txt"
    p.write_text(content, encoding="utf-8")

def cache_get(key: str) -> t.Optional[str]:
    p = CACHE_DIR / f"{key}.txt"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None
    return None

# ============
# PIPELINE ANALYSE
# ============
def run_direct_analysis_stream(base_url: str, model: str, prompt_template: str, text: str,
                               temperature: float, num_ctx: int, top_p: float) -> t.Iterator[str]:
    prompt = prompt_template.replace("{DOCUMENT}", text)
    yield f"⏳ Analyse directe avec **{model}** …\n"
    for chunk in ollama_generate_stream(
        base_url, model, prompt, temperature=temperature, num_ctx=num_ctx, top_p=top_p
    ):
        yield chunk

def run_chunked_analysis_stream(base_url: str, model_chunk: str, model_agg: str,
                                chunk_prompt: str, agg_prompt: str, text: str,
                                chunk_max: int, chunk_overlap: int,
                                temperature: float, num_ctx: int, top_p: float) -> t.Iterator[str]:
    chunks = chunk_text(text, max_chars=chunk_max, overlap=chunk_overlap)
    n = len(chunks)
    if n == 0:
        yield "⚠️ Aucun chunk (texte vide)."
        return
    summaries = []
    yield f"🧩 Mode **chunks** : {n} morceaux …\n"
    for i, ch in enumerate(chunks, 1):
        prompt_i = (chunk_prompt
                    .replace("{I}", str(i))
                    .replace("{N}", str(n))
                    .replace("{CHUNK}", ch))
        yield f"\n---\n**Chunk {i}/{n} — {len(ch)} caractères**\n"
        cur = []
        for piece in ollama_generate_stream(
            base_url, model_chunk, prompt_i, temperature=temperature, num_ctx=num_ctx, top_p=top_p
        ):
            cur.append(piece)
            yield piece
        summaries.append("".join(cur))

    # Agrégation
    summaries_blob = "\n\n---\n".join(summaries)
    prompt_agg = agg_prompt.replace("{SUMMARIES}", summaries_blob)
    yield "\n\n🧷 **Agrégation** …\n"
    for piece in ollama_generate_stream(
        base_url, model_agg, prompt_agg, temperature=temperature, num_ctx=num_ctx, top_p=top_p
    ):
        yield piece

# ============
# GRADIO CALLBACKS
# ============
def refresh_models(ollama_url: str):
    models, msg = list_models(ollama_url)
    model_default = models[0] if models else ""
    return gr.update(choices=models, value=model_default), gr.update(choices=models, value=model_default), gr.update(choices=models, value=model_default), msg

def handle_file(file, force_ocr: bool, ocr_lang: str, do_clean: bool, do_anonym: bool):
    if not file:
        return "❌ Aucun fichier", "", "", 0
    p = Path(file.name)
    tmp = CACHE_DIR / sanitize_filename(p.name)
    shutil.copyfile(file.name, tmp)

    fhash = sha256_file(tmp)
    cache_raw = CACHE_DIR / f"{fhash}.raw.txt"
    if cache_raw.exists() and force_ocr is False:
        raw = cache_raw.read_text(encoding="utf-8", errors="ignore")
        status = f"♻️ Texte brut chargé depuis cache (hash {fhash[:8]})."
    else:
        raw = extract_text(tmp, force_ocr=force_ocr, ocr_lang=ocr_lang)
        cache_raw.write_text(raw, encoding="utf-8")
        status = f"✅ Extraction (hash {fhash[:8]}) — force_ocr={force_ocr}, ocr_lang={ocr_lang}"

    pre = preprocess_text(raw, do_clean=do_clean, do_anonym=do_anonym)
    cache_pre = CACHE_DIR / f"{fhash}.pre.txt"
    cache_pre.write_text(pre, encoding="utf-8")

    return status, fhash, pre, len(pre)

def analyze(
    mode_choice: str,
    ollama_url: str,
    model_direct: str,
    model_chunk: str,
    model_agg: str,
    text_input: str,
    direct_prompt: str,
    chunk_prompt: str,
    agg_prompt: str,
    temp: float,
    top_p: float,
    num_ctx: int,
    chunk_max: int,
    chunk_overlap: int,
    file_hash: str,
    do_clean: bool,
    do_anonym: bool,
    force_ocr: bool,
    ocr_lang: str,
    max_direct_chars: int,
):
    """
    Stream des résultats + gestion cache.
    """
    if not text_input.strip():
        yield "❌ Aucun texte à analyser."
        return

    # Choix mode auto selon seuil si demandé
    mode = mode_choice
    if mode_choice == "Auto":
        mode = "Direct" if len(text_input) <= max_direct_chars else "Chunks"

    # Clé de cache
    key = cache_key_for(
        file_hash or "NOFILE",
        do_clean, do_anonym, force_ocr, ocr_lang,
        "direct" if mode == "Direct" else "chunks",
        model_direct, model_chunk, model_agg,
        direct_prompt, chunk_prompt, agg_prompt,
        chunk_max, chunk_overlap
    )
    cached = cache_get(key)
    if cached:
        yield f"♻️ Résultat depuis cache (clé {key[:10]})\n\n{cached}"
        return

    # Pas de cache -> on stream & on enregistre
    collected = []
    def sink(s: str):
        collected.append(s)
        yield s

    header = f"**Mode**: {mode} — **Temp**: {temp} — **Top-p**: {top_p} — **num_ctx**: {num_ctx}\n"
    yield header

    if mode == "Direct":
        for piece in run_direct_analysis_stream(
            ollama_url, model_direct, direct_prompt, text_input, temp, num_ctx, top_p
        ):
            for y in sink(piece):
                yield y
    else:
        # Chunks
        for piece in run_chunked_analysis_stream(
            ollama_url, model_chunk, model_agg, chunk_prompt, agg_prompt,
            text_input, chunk_max, chunk_overlap, temp, num_ctx, top_p
        ):
            for y in sink(piece):
                yield y

    final = "".join(collected).strip()
    if final:
        cache_put(key, final)

# ============
# UI GRADIO
# ============
with gr.Blocks(title="OCR + Analyse (Ollama on Runpod)") as demo:
    gr.Markdown("## 🧠 OCR + Analyse (Ollama) — Pod Runpod\n"
                "- Endpoint par défaut : `https://tszwlwj46df2kj-11434.proxy.runpod.net`\n"
                "- Rafraîchissez la liste des modèles, uploadez un document, puis lancez l'analyse.\n"
                "- Streaming activé ; cache automatique par hash et paramètres.\n")

    with gr.Tab("Source & Prétraitement"):
        with gr.Row():
            with gr.Column(scale=1):
                ollama_url = gr.Textbox(label="Endpoint Ollama", value=DEFAULT_OLLAMA_URL)
                refresh_btn = gr.Button("🔄 Rafraîchir modèles")
                models_status = gr.Textbox(label="Statut modèles", value="", interactive=False)

                model_direct = gr.Dropdown(label="Modèle (Direct)", choices=[], value=None)
                model_chunk  = gr.Dropdown(label="Modèle (Chunks)", choices=[], value=None)
                model_agg    = gr.Dropdown(label="Modèle (Agrégation)", choices=[], value=None)

            with gr.Column(scale=2):
                file = gr.File(label="Fichier (PDF, TXT, DOCX, PNG/JPG)")
                force_ocr = gr.Checkbox(label="Forcer OCR", value=False)
                ocr_lang  = gr.Textbox(label="Langue OCR (tesseract)", value="fra")
                do_clean  = gr.Checkbox(label="Nettoyage", value=True)
                do_anonym = gr.Checkbox(label="Anonymisation", value=False)
                preprocess_btn = gr.Button("📄 Extraire & Prétraiter")
                status = gr.Textbox(label="Statut", value="", interactive=False)

        with gr.Row():
            file_hash_box = gr.Textbox(label="Hash fichier", value="", interactive=False)
            text_pre = gr.Textbox(label="Texte (prétraité)", lines=16)
            text_len = gr.Number(label="Longueur (caractères)", value=0, precision=0)

    with gr.Tab("Prompts & Modèles"):
        with gr.Row():
            with gr.Column():
                mode_choice = gr.Radio(
                    choices=["Auto", "Direct", "Chunks"],
                    value="Auto",
                    label="Mode d'analyse"
                )
                max_direct_chars = gr.Slider(2000, 30000, value=DEFAULT_MAX_DIRECT_CHARS, step=500, label="Seuil Direct (Auto)")
                chunk_max = gr.Slider(2000, 8000, value=DEFAULT_CHUNK_MAX_CHARS, step=100, label="Taille chunk")
                chunk_overlap = gr.Slider(0, 1000, value=DEFAULT_CHUNK_OVERLAP, step=50, label="Overlap")

                temp = gr.Slider(0, 1, value=DEFAULT_TEMPERATURE, step=0.05, label="Température")
                top_p = gr.Slider(0, 1, value=0.9, step=0.05, label="Top‑p")
                num_ctx = gr.Slider(1024, 32768, value=DEFAULT_NUM_CTX, step=512, label="num_ctx")
            with gr.Column():
                direct_prompt = gr.Textbox(label="Prompt (Direct)", value=DEFAULT_DIRECT_PROMPT, lines=10)
                chunk_prompt  = gr.Textbox(label="Prompt (Chunk)", value=DEFAULT_CHUNK_PROMPT, lines=8)
                agg_prompt    = gr.Textbox(label="Prompt (Agrégation)", value=DEFAULT_AGG_PROMPT, lines=8)

    with gr.Tab("Analyse"):
        with gr.Row():
            analyze_btn = gr.Button("🚀 Lancer l'analyse (streaming)", variant="primary")
        result = gr.Textbox(label="Résultat", lines=28)

    # ====== Événements
    refresh_btn.click(
        fn=refresh_models,
        inputs=[ollama_url],
        outputs=[model_direct, model_chunk, model_agg, models_status]
    )

    preprocess_btn.click(
        fn=handle_file,
        inputs=[file, force_ocr, ocr_lang, do_clean, do_anonym],
        outputs=[status, file_hash_box, text_pre, text_len]
    )

    analyze_btn.click(
        fn=analyze,
        inputs=[
            mode_choice, ollama_url,
            model_direct, model_chunk, model_agg,
            text_pre,
            direct_prompt, chunk_prompt, agg_prompt,
            temp, top_p, num_ctx,
            chunk_max, chunk_overlap,
            file_hash_box, do_clean, do_anonym, force_ocr, ocr_lang, max_direct_chars
        ],
        outputs=[result]
    )

# File d'attente (Cloudflare/proxy-friendly)
demo = demo.queue(concurrency_count=2)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

