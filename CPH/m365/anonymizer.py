#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anonymiseur PDF avec OCR auto si pas de texte + mode de désanonymisation (CLI).
- Mode 'anonymize':
  * Si le PDF a peu/pas de texte -> OCR (ocrmypdf --skip-text) pour créer un PDF "searchable".
  * Extrait le texte et applique des remplacements par catégories (configurables).
  * Produit:
      1) Un PDF "anonymized_redacted.pdf" avec de vraies redactions (irréversible, pour diffusion).
      2) Un fichier "anonymized.txt" (réversible via placeholders).
      3) Un "mapping.json" (correspondances original -> placeholder) pour désanonymiser.
      4) Un "original_text.txt" (optionnel).
      5) Un "used_config.json" (trace des préférences utilisées).
- Mode 'deanonymize':
  * Prend un texte (fichier .txt ou du texte extrait d’un PDF contenant les placeholders) + mapping.json
  * Reconstruit un "deanonymized.txt".
  * (Note: un PDF redacted est irréversible par définition)
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

# -------------------------
# Configuration & RegEx
# -------------------------

DEFAULT_CONFIG = {
    "categories": {
        "email": True,
        "phone_fr": True,
        "iban_fr": True,
        "siren": True,
        "siret": True,
        "amount_eur": True,
        "dates": True,
        "person_names": True,
        "company_names": True,
        "addresses_fr": True
    },
    "custom_entities": [],  # liste d'expressions à anonymiser systématiquement (regex ou littéral)
    "output": {
        "emit_original_text": True
    },
    "ocr": {
        "enable": True,
        "languages": "fra+eng",
        "min_avg_chars_per_page": 15,  # si texte < 15 chars/page en moyenne -> OCR
        "force_ocr": False  # si True: OCR sur toutes les pages (avec ocrmypdf)
    }
}

# Regex patterns
PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone_fr": re.compile(r"\b(?:\+33|0)\s*(?:1-9{4})\b"),
    "iban_fr": re.compile(r"\bFR\d{2}(?:[\s]?\d{4}){5}\s?\d{2}\b", re.IGNORECASE),
    "siret": re.compile(r"\b\d{3}\s?\d{3}\s?\d{3}\s?\d{5}\b"),
    "siren": re.compile(r"\b\d{3}\s?\d{3}\s?\d{3}\b(?!\s?\d{5})"),
    "amount_eur": re.compile(
        r"\b(?:€\s?\d{1,3}(?:[ .]\d{3})*(?:,\d{2})?|\d{1,3}(?:[ .]\d{3})*(?:,\d{2})?\s?(?:€|euros?))\b",
        re.IGNORECASE
    ),
    # Dates: 01/02/2023, 01-02-2023, 1er janvier 2023, 12 mars 21, etc.
    "dates": re.compile(
        r"\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}\s?(?:janv\.?|févr\.?|mars|avr\.?|mai|juin|juil\.?|août|sept\.?|oct\.?|nov\.?|déc\.?|janvier|février|avril|juillet|septembre|octobre|novembre|décembre)\s?\d{2,4}|(?:\d{1,2})(?:er)?\s?(?:janv\.?|févr\.?|mars|avr\.?|mai|juin|juil\.?|août|sept\.?|oct\.?|nov\.?|déc\.)\s?\d{2,4})\b",
        re.IGNORECASE
    ),
    # Personnes: titres (M., Mme, Monsieur, Madame) + nom/prénom (heuristique)
    "person_names": re.compile(
        r"\b(?:M\.|Mme|Monsieur|Madame)\s+[A-ZÉÈÊÂÎÔÛÀÇ][A-Za-zÀ-ÖØ-öø-ÿ'’\- ]{1,40}\b"
    ),
    # Sociétés: formes juridiques + dénomination
    "company_names": re.compile(
        r"\b(?:SARL|SASU?|SA|EURL|SCI|SNC|CCI|Association)\s+[A-Z0-9][A-Z0-9 &'’\-]{1,60}\b"
    ),
    # Adresses FR: numéro + type + voie (+ CP possible ailleurs)
    "addresses_fr": re.compile(
        r"\b\d{1,4}\s+(?:rue|avenue|av\.?|boulevard|bd\.?|chemin|allée|impasse|route|quai|cours|place|pl\.?)\s+[A-Za-zÀ-ÖØ-öø-ÿ'’\-\s]{2,}\b",
        re.IGNORECASE
    ),
    "postal_code_fr": re.compile(r"\b\d{5}\b")
}

PLACEHOLDER_PREFIX = {
    "email": "EMAIL",
    "phone_fr": "TEL",
    "iban_fr": "IBAN",
    "siren": "SIREN",
    "siret": "SIRET",
    "amount_eur": "AMOUNT",
    "dates": "DATE",
    "person_names": "NAME",
    "company_names": "COMPANY",
    "addresses_fr": "ADDRESS",
    "postal_code_fr": "CP",
    "custom": "CUSTOM"
}

@dataclass
class AnonymizerConfig:
    categories: Dict[str, bool]
    custom_entities: List[str]
    output: Dict[str, bool]
    ocr: Dict[str, object]

    @staticmethod
    def from_file(path: Path) -> "AnonymizerConfig":
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = DEFAULT_CONFIG

        def deep_merge(default, user):
            if isinstance(default, dict):
                out = dict(default)
                for k, v in (user or {}).items():
                    out[k] = deep_merge(default.get(k), v) if isinstance(v, dict) else v
                return out
            return user if user is not None else default

        merged = deep_merge(DEFAULT_CONFIG, data)
        return AnonymizerConfig(**merged)

# -------------------------
# Utils
# -------------------------

def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def run_ocr_if_needed(input_pdf: Path, cfg: AnonymizerConfig, workdir: Path) -> Path:
    """Retourne un PDF 'searchable'. Si déjà textuel, renvoie l'original.
       Sinon, utilise ocrmypdf (--skip-text)."""
    try:
        doc = fitz.open(input_pdf)
    except Exception as e:
        raise RuntimeError(f"Impossible d'ouvrir le PDF: {e}")

    total_chars = 0
    for page in doc:
        try:
            txt = page.get_text("text") or ""
        except Exception:
            txt = ""
        total_chars += len(txt.strip())
    avg = total_chars / max(1, len(doc))
    doc.close()

    if not cfg.ocr.get("enable", True):
        logging.info("OCR désactivé par la configuration.")
        return input_pdf

    if cfg.ocr.get("force_ocr", False) or avg < int(cfg.ocr.get("min_avg_chars_per_page", 15)):
        logging.info(f"OCR requis (moyenne {avg:.1f} chars/page).")
        ocrmypdf = shutil.which("ocrmypdf")
        if not ocrmypdf:
            logging.warning("ocrmypdf introuvable. Le PDF restera non OCRisé.")
            return input_pdf
        output_pdf = workdir / f"{input_pdf.stem}_ocr.pdf"
        cmd = [
            ocrmypdf,
            "--skip-text" if not cfg.ocr.get("force_ocr", False) else "",
            "--language", str(cfg.ocr.get("languages", "fra+eng")),
            "--output-type", "pdf",
            str(input_pdf), str(output_pdf)
        ]
        cmd = [c for c in cmd if c != ""]
        logging.info(f"Exécution: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_pdf
        except subprocess.CalledProcessError as e:
            logging.error(f"OCR échoué: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)}")
            return input_pdf
    else:
        logging.info(f"OCR non nécessaire (moyenne {avg:.1f} chars/page).")
        return input_pdf

def prepare_output_paths(input_pdf: Path, outdir: Path) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    base = input_pdf.stem
    return {
        "redacted_pdf": outdir / f"{base}_anonymized_redacted.pdf",
        "anonymized_txt": outdir / f"{base}_anonymized.txt",
        "original_txt": outdir / f"{base}_original_text.txt",
        "mapping_json": outdir / f"{base}_mapping.json",
        "used_config": outdir / f"{base}_used_config.json",
        "deanonymized_txt": outdir / f"{base}_deanonymized.txt",
    }

def build_custom_patterns(custom_entities: List[str]) -> List[Tuple[str, re.Pattern]]:
    pats = []
    for i, expr in enumerate(custom_entities or []):
        try:
            if expr.startswith("/") and expr.endswith("/"):
                pat = re.compile(expr[1:-1], re.IGNORECASE)
            else:
                pat = re.compile(re.escape(expr), re.IGNORECASE)
            pats.append((f"custom_{i+1}", pat))
        except re.error:
            logging.warning(f"Regex invalide ignorée: {expr}")
    return pats

# -------------------------
# Anonymization Core
# -------------------------

class Anonymizer:
    def __init__(self, cfg: AnonymizerConfig):
        self.cfg = cfg
        self.counters = {k: 0 for k in PLACEHOLDER_PREFIX}
        self.mapping: Dict[str, Dict[str, str]] = {}  # original -> {type, placeholder}

    def _next_placeholder(self, typ: str) -> str:
        self.counters[typ] += 1
        return f"⟦{PLACEHOLDER_PREFIX.get(typ, 'X')}_{self.counters[typ]:03d}⟧"

    def _register(self, typ: str, original: str) -> str:
        key = original
        if key in self.mapping:
            return self.mapping[key]["placeholder"]
        ph = self._next_placeholder(typ)
        self.mapping[key] = {"type": typ, "placeholder": ph}
        return ph

    def _collect_matches(self, text: str) -> List[Tuple[int, int, str, str]]:
        candidates: List[Tuple[int, int, str, str]] = []

        for typ, enabled in self.cfg.categories.items():
            if not enabled:
                continue
            pat = PATTERNS.get(typ)
            if not pat:
                continue
            for m in pat.finditer(text):
                s, e = m.span()
                candidates.append((s, e, text[s:e], typ))

        if self.cfg.categories.get("addresses_fr", True):
            for m in PATTERNS["postal_code_fr"].finditer(text):
                s, e = m.span()
                candidates.append((s, e, text[s:e], "postal_code_fr"))

        for name, pat in build_custom_patterns(self.cfg.custom_entities):
            for m in pat.finditer(text):
                s, e = m.span()
                candidates.append((s, e, text[s:e], "custom"))

        candidates.sort(key=lambda x: (x[0], -(x[1]-x[0])))
        resolved: List[Tuple[int, int, str, str]] = []
        last_end = -1
        for s, e, orig, typ in candidates:
            if s >= last_end:
                resolved.append((s, e, orig, typ))
                last_end = e
            else:
                ps, pe, po, pt = resolved[-1]
                if (e - s) > (pe - ps):
                    resolved[-1] = (s, e, orig, typ)
                    last_end = e
        return resolved

    def anonymize_text(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        matches = self._collect_matches(text)
        if not matches:
            return text, []
        out = []
        last = 0
        replaced_pairs: List[Tuple[str, str]] = []
        for s, e, original, typ in matches:
            out.append(text[last:s])
            placeholder = self._register(typ, original)
            out.append(placeholder)
            replaced_pairs.append((original, placeholder))
            last = e
        out.append(text[last:])
        return "".join(out), replaced_pairs

# -------------------------
# PDF Processing
# -------------------------

def anonymize_pdf(input_pdf: Path, outdir: Path, cfg_path: Path = None):
    cfg = AnonymizerConfig.from_file(cfg_path) if cfg_path else AnonymizerConfig(**DEFAULT_CONFIG)
    outpaths = prepare_output_paths(input_pdf, outdir)
    workdir = Path(tempfile.mkdtemp(prefix="anon_"))

    searchable_pdf = run_ocr_if_needed(input_pdf, cfg, workdir)

    doc = fitz.open(searchable_pdf)
    anonymizer = Anonymizer(cfg)
    all_original_text = []
    all_anonymized_text = []

    per_page_originals: List[Dict[str, List[str]]] = []

    for i, page in enumerate(doc):
        page_text = page.get_text("text") or ""
        all_original_text.append(page_text)

        anon_text, pairs = anonymizer.anonymize_text(page_text)
        all_anonymized_text.append(anon_text)

        originals_this_page: Dict[str, List[str]] = {}
        for original, placeholder in pairs:
            originals_this_page.setdefault(original, []).append(placeholder)
        per_page_originals.append(originals_this_page)
    doc.close()

    # ---- 1) PDF redacted (irréversible) ----
    redacted_doc = fitz.open(searchable_pdf)  # nouvelle instance
    redacted_count = 0
    for page_idx, page in enumerate(redacted_doc):
        originals = list(per_page_originals[page_idx].keys())
        page_added = False
        for original in set(originals):
            try:
                rects = page.search_for(original)  # case-sensitive
            except Exception:
                rects = []
            for r in rects:
                page.add_redact_annot(r, fill=(0, 0, 0))
                redacted_count += 1
                page_added = True
        if page_added:
            page.apply_redactions()

    redacted_doc.save(outpaths["redacted_pdf"])
    redacted_doc.close()

    # ---- 2) Texte anonymisé + mapping ----
    anonymized_text_joined = "\n".join(all_anonymized_text)
    with open(outpaths["anonymized_txt"], "w", encoding="utf-8") as f:
        f.write(anonymized_text_joined)

    if cfg.output.get("emit_original_text", True):
        with open(outpaths["original_txt"], "w", encoding="utf-8") as f:
            f.write("\n".join(all_original_text))

    mapping_payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_file": str(input_pdf.name),
        "searchable_source": str(searchable_pdf.name) if searchable_pdf != input_pdf else None,
        "doc_sha256": compute_file_hash(searchable_pdf),
        "placeholders": anonymizer.mapping,  # original -> {type, placeholder}
        "counters": anonymizer.counters,
        "config": asdict(cfg)
    }
    with open(outpaths["mapping_json"], "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, ensure_ascii=False, indent=2)

    with open(outpaths["used_config"], "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    logging.info(f"Anonymisation terminée. Redactions posées: {redacted_count}")
    logging.info(f"Sorties:\n- {outpaths['redacted_pdf']}\n- {outpaths['anonymized_txt']}\n- {outpaths['mapping_json']}")
    return outpaths

def deanonymize_text(input_text_path: Path, mapping_path: Path, output_path: Path):
    with open(input_text_path, "r", encoding="utf-8") as f:
        text = f.read()
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_payload = json.load(f)
    inv = {}
    for original, info in mapping_payload.get("placeholders", {}).items():
        ph = info["placeholder"]
        inv.setdefault(ph, original)

    placeholders_sorted = sorted(inv.keys(), key=len, reverse=True)
    for ph in placeholders_sorted:
        text = text.replace(ph, inv[ph])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    logging.info(f"Texte désanonymisé écrit dans {output_path}")
    return output_path

# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Anonymisation PDF avec OCR auto et désanonymisation via mapping.")
    parser.add_argument("--mode", choices=["anonymize", "deanonymize"], required=True, help="Mode d'exécution.")
    parser.add_argument("--input", required=True, help="Fichier d'entrée (PDF en anonymize, TXT/PDF en deanonymize).")
    parser.add_argument("--outdir", default="out", help="Dossier de sortie.")
    parser.add_argument("--config", help="Fichier JSON de configuration des catégories/règles.")
    parser.add_argument("--mapping", help="Fichier de mapping (obligatoire en mode deanonymize).")
    parser.add_argument("--log", default="INFO", help="Niveau de log (DEBUG, INFO, WARNING, ERROR).")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(levelname)s - %(message)s")

    outdir = Path(args.outdir)
    if args.mode == "anonymize":
        input_pdf = Path(args.input)
        if not input_pdf.exists():
            logging.error(f"Fichier introuvable: {input_pdf}")
            sys.exit(1)
        anonymize_pdf(input_pdf, outdir, Path(args.config) if args.config else None)

    elif args.mode == "deanonymize":
        if not args.mapping:
            logging.error("Vous devez fournir --mapping en mode deanonymize.")
            sys.exit(1)
        input_path = Path(args.input)
        mapping_path = Path(args.mapping)
        if not input_path.exists():
            logging.error(f"Fichier introuvable: {input_path}")
            sys.exit(1)
        if not mapping_path.exists():
            logging.error(f"Mapping introuvable: {mapping_path}")
            sys.exit(1)

        outpaths = prepare_output_paths(input_path, Path(args.outdir))
        # Si input est un PDF, on extrait son texte d'abord
        if input_path.suffix.lower() == ".pdf":
            doc = fitz.open(input_path)
            txt = "\n".join([(p.get_text("text") or "") for p in doc])
            doc.close()
            tmp_txt = outpaths["anonymized_txt"].with_name(f"{input_path.stem}_text_for_deanonymize.txt")
            with open(tmp_txt, "w", encoding="utf-8") as f:
                f.write(txt)
            deanonymize_text(tmp_txt, mapping_path, outpaths["deanonymized_txt"])
        else:
            deanonymize_text(input_path, mapping_path, outpaths["deanonymized_txt"])

if __name__ == "__main__":
    main()

