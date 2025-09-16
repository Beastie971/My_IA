#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import fitz
import hashlib
import logging
import tempfile
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple

import gradio as gr

# =========================
# ====== Anonymiseur ======
# =========================

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
    "custom_entities": [],  # regex /.../ ou texte littéral
    "output": {
        "emit_original_text": True
    },
    "ocr": {
        "enable": True,
        "languages": "fra+eng",
        "min_avg_chars_per_page": 15,
        "force_ocr": False
    }
}

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
    "dates": re.compile(
        r"\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}\s?(?:janv\.?|févr\.?|mars|avr\.?|mai|juin|juil\.?|août|sept\.?|oct\.?|nov\.?|déc\.?|janvier|février|avril|juillet|septembre|octobre|novembre|décembre)\s?\d{2,4}|(?:\d{1,2})(?:er)?\s?(?:janv\.?|févr\.?|mars|avr\.?|mai|juin|juil\.?|août|sept\.?|oct\.?|nov\.?|déc\.)\s?\d{2,4})\b",
        re.IGNORECASE
    ),
    "person_names": re.compile(
        r"\b(?:M\.|Mme|Monsieur|Madame)\s+[A-ZÉÈÊÂÎÔÛÀÇ][A-Za-zÀ-ÖØ-öø-ÿ'’\- ]{1,40}\b"
    ),
    "company_names": re.compile(
        r"\b(?:SARL|SASU?|SA|EURL|SCI|SNC|CCI|Association)\s+[A-Z0-9][A-Z0-9 &'’\-]{1,60}\b"
    ),
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
    def from_dict(d: dict) -> "AnonymizerConfig":
        # merge deep with defaults
        def deep_merge(default, user):
            if isinstance(default, dict):
                out = dict(default)
                for k, v in (user or {}).items():
                    out[k] = deep_merge(default.get(k), v) if isinstance(v, dict) else v
                return out
            return user if user is not None else default
        merged = deep_merge(DEFAULT_CONFIG, d or {})
        return AnonymizerConfig(**merged)

def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_custom_patterns(custom_entities: List[str]) -> List[Tuple[str, re.Pattern]]:
    pats = []
    for i, expr in enumerate(custom_entities or []):
        try:
            if isinstance(expr, str) and expr.startswith("/") and expr.endswith("/"):
                pat = re.compile(expr[1:-1], re.IGNORECASE)
            else:
                pat = re.compile(re.escape(str(expr)), re.IGNORECASE)
            pats.append((f"custom_{i+1}", pat))
        except re.error:
            pass
    return pats

class Anonymizer:
    def __init__(self, cfg: AnonymizerConfig):
        self.cfg = cfg
        self.counters = {k: 0 for k in PLACEHOLDER_PREFIX}
        self.mapping: Dict[str, Dict[str, str]] = {}

    def _next_placeholder(self, typ: str) -> str:
        self.counters[typ] += 1
        return f"⟦{PLACEHOLDER_PREFIX.get(typ, 'X')}_{self.counters[typ]:03d}⟧"

    def _register(self, typ: str, original: str) -> str:
        if original in self.mapping:
            return self.mapping[original]["placeholder"]
        ph = self._next_placeholder(typ)
        self.mapping[original] = {"type": typ, "placeholder": ph}
        return ph

    def _collect_matches(self, text: str) -> List[Tuple[int, int, str, str]]:
        cands: List[Tuple[int, int, str, str]] = []
        for typ, enabled in self.cfg.categories.items():
            if not enabled: 
                continue
            pat = PATTERNS.get(typ)
            if not pat: 
                continue
            for m in pat.finditer(text):
                s, e = m.span()
                cands.append((s, e, text[s:e], typ))
        if self.cfg.categories.get("addresses_fr", True):
            for m in PATTERNS["postal_code_fr"].finditer(text):
                s, e = m.span()
                cands.append((s, e, text[s:e], "postal_code_fr"))
        for name, pat in build_custom_patterns(self.cfg.custom_entities):
            for m in pat.finditer(text):
                s, e = m.span()
                cands.append((s, e, text[s:e], "custom"))
        cands.sort(key=lambda x: (x[0], -(x[1]-x[0])))
        resolved, last_end = [], -1
        for s, e, o, t in cands:
            if s >= last_end:
                resolved.append((s, e, o, t))
                last_end = e
            else:
                ps, pe, po, pt = resolved[-1]
                if (e - s) > (pe - ps):
                    resolved[-1] = (s, e, o, t)
                    last_end = e
        return resolved

    def anonymize_text(self, text: str):
        matches = self._collect_matches(text)
        if not matches:
            return text, []
        out, last = [], 0
        pairs = []
        for s, e, original, typ in matches:
            out.append(text[last:s])
            ph = self._register(typ, original)
            out.append(ph)
            pairs.append((original, ph))
            last = e
        out.append(text[last:])
        return "".join(out), pairs

def run_ocr_if_needed(input_pdf: Path, cfg: AnonymizerConfig, workdir: Path, log: List[str]) -> Path:
    try:
        doc = fitz.open(input_pdf)
    except Exception as e:
        raise RuntimeError(f"Impossible d'ouvrir le PDF: {e}")
    total_chars = 0
    for p in doc:
        try:
            txt = p.get_text("text") or ""
        except Exception:
            txt = ""
        total_chars += len(txt.strip())
    avg = total_chars / max(1, len(doc))
    doc.close()

    if not cfg.ocr.get("enable", True):
        log.append("OCR désactivé par la configuration.")
        return input_pdf

    ocrmypdf = shutil.which("ocrmypdf")
    if (cfg.ocr.get("force_ocr", False) or avg < int(cfg.ocr.get("min_avg_chars_per_page", 15))):
        if not ocrmypdf:
            log.append("⚠️ ocrmypdf introuvable. Le document ne sera pas OCRisé.")
            return input_pdf
        outpdf = workdir / f"{input_pdf.stem}_ocr.pdf"
        cmd = [ocrmypdf]
        if not cfg.ocr.get("force_ocr", False):
            cmd += ["--skip-text"]
        cmd += ["--language", str(cfg.ocr.get("languages", "fra+eng")), "--output-type", "pdf", str(input_pdf), str(outpdf)]
        log.append("Exécution OCR: " + " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            log.append("✅ OCR terminé.")
            return outpdf
        except subprocess.CalledProcessError as e:
            log.append("❌ OCR échoué: " + (e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)))
            return input_pdf
    else:
        log.append(f"OCR non nécessaire (moyenne {avg:.1f} char/page).")
        return input_pdf

def anonymize_pdf_gradio(input_pdf_path: str, cfg_dict: dict, outdir: Path) -> Dict[str, str]:
    log = []
    cfg = AnonymizerConfig.from_dict(cfg_dict)
    workdir = Path(tempfile.mkdtemp(prefix="anon_"))
    outdir.mkdir(parents=True, exist_ok=True)

    input_pdf = Path(input_pdf_path)
    base = input_pdf.stem

    def out(name): return outdir / f"{base}_{name}"

    try:
        searchable = run_ocr_if_needed(input_pdf, cfg, workdir, log)
        doc = fitz.open(searchable)
        anonymizer = Anonymizer(cfg)

        all_original_text, all_anonymized_text = [], []
        per_page_originals = []

        for page in doc:
            t = page.get_text("text") or ""
            all_original_text.append(t)
            anon_t, pairs = anonymizer.anonymize_text(t)
            all_anonymized_text.append(anon_t)
            originals = {}
            for original, placeholder in pairs:
                originals.setdefault(original, []).append(placeholder)
            per_page_originals.append(originals)
        doc.close()

        # Redactions
        redacted_doc = fitz.open(searchable)
        redacted_count = 0
        for idx, page in enumerate(redacted_doc):
            for original in set(per_page_originals[idx].keys()):
                try:
                    rects = page.search_for(original)
                except Exception:
                    rects = []
                for r in rects:
                    page.add_redact_annot(r, fill=(0, 0, 0))
                    redacted_count += 1
            if page.has_redactions():
                page.apply_redactions()
        redacted_pdf = out("anonymized_redacted.pdf")
        redacted_doc.save(redacted_pdf)
        redacted_doc.close()
        log.append(f"🧽 Redactions posées: {redacted_count}")

        # Texte + mapping
        anonymized_txt = out("anonymized.txt")
        with open(anonymized_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(all_anonymized_text))

        mapping_json = out("mapping.json")
        mapping_payload = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_file": input_pdf.name,
            "searchable_source": searchable.name if searchable != input_pdf else None,
            "doc_sha256": compute_file_hash(searchable),
            "placeholders": anonymizer.mapping,
            "counters": anonymizer.counters,
            "config": asdict(cfg)
        }
        with open(mapping_json, "w", encoding="utf-8") as f:
            json.dump(mapping_payload, f, ensure_ascii=False, indent=2)

        used_config = out("used_config.json")
        with open(used_config, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

        original_txt = None
        if cfg.output.get("emit_original_text", True):
            original_txt = out("original_text.txt")
            with open(original_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(all_original_text))

        # Aperçu replacements
        preview_lines = []
        for k, v in list(anonymizer.mapping.items())[:20]:
            preview_lines.append(f"- `{k}` → **{v['placeholder']}** ({v['type']})")
        preview = "Aperçu des 20 premiers remplacements :\n" + ("\n".join(preview_lines) if preview_lines else "_Aucun motif détecté._")

        return {
            "redacted_pdf": str(redacted_pdf),
            "anonymized_txt": str(anonymized_txt),
            "mapping_json": str(mapping_json),
            "used_config": str(used_config),
            "original_txt": str(original_txt) if original_txt else "",
            "log": "\n".join(log),
            "preview": preview
        }
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass

def deanonymize_file_gradio(input_path: str, mapping_path: str, outdir: Path) -> Dict[str, str]:
    log = []
    input_p = Path(input_path)
    mapping_p = Path(mapping_path)
    outdir.mkdir(parents=True, exist_ok=True)
    base = input_p.stem

    def out(name): return outdir / f"{base}_{name}"

    if not mapping_p.exists():
        raise RuntimeError("Mapping introuvable.")

    # Extraire texte si PDF
    if input_p.suffix.lower() == ".pdf":
        doc = fitz.open(input_p)
        text = "\n".join([p.get_text("text") or "" for p in doc])
        doc.close()
        txt_path = out("text_for_deanonymize.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        src_text_path = txt_path
        log.append("Texte extrait du PDF pour désanonymisation.")
    else:
        src_text_path = input_p

    with open(src_text_path, "r", encoding="utf-8") as f:
        text = f.read()
    with open(mapping_p, "r", encoding="utf-8") as f:
        mapping_payload = json.load(f)

    inv = {}
    for original, info in mapping_payload.get("placeholders", {}).items():
        ph = info["placeholder"]
        if ph not in inv:
            inv[ph] = original

    # Remplacement (plus longs d'abord)
    for ph in sorted(inv.keys(), key=len, reverse=True):
        text = text.replace(ph, inv[ph])

    dean_txt = out("deanonymized.txt")
    with open(dean_txt, "w", encoding="utf-8") as f:
        f.write(text)

    log.append("✅ Désanonymisation terminée (fichier texte).")
    log.append("⚠️ Le PDF redacted est irréversible par conception.")

    return {
        "deanonymized_txt": str(dean_txt),
        "log": "\n".join(log)
    }

# =========================
# ===== Interface UI =====
# =========================

def default_cfg_dict():
    return json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

def build_ui():
    with gr.Blocks(title="Anonymiseur PDF avec OCR & Désanonymisation", theme="soft") as demo:
        gr.Markdown("## 🛡️ Anonymiseur PDF (OCR auto si besoin) + Désanonymisation\n"
                    "Confidentialité: tout se passe **localement**. Aucun envoi de données.")

        with gr.Row():
            mode = gr.Radio(choices=["Anonymiser", "Désanonymiser"], value="Anonymiser", label="Mode")

        with gr.Row():
            input_pdf = gr.File(label="PDF à traiter", file_types=[".pdf"], visible=True)
            input_for_dean = gr.File(label="Fichier anonymisé (TXT ou PDF) pour désanonymisation", visible=False)
            mapping_in = gr.File(label="mapping.json (obligatoire pour désanonymisation)", file_types=[".json"], visible=False)

        with gr.Accordion("⚙️ Préférences d’anonymisation / OCR", open=False):
            with gr.Row():
                email = gr.Checkbox(True, label="Emails")
                phone_fr = gr.Checkbox(True, label="Téléphones FR")
                iban_fr = gr.Checkbox(True, label="IBAN FR")
                siren = gr.Checkbox(True, label="SIREN")
                siret = gr.Checkbox(True, label="SIRET")
            with gr.Row():
                amount_eur = gr.Checkbox(True, label="Montants €")
                dates = gr.Checkbox(True, label="Dates")
                person_names = gr.Checkbox(True, label="Noms (heuristique)")
                company_names = gr.Checkbox(True, label="Dénominations sociales")
                addresses_fr = gr.Checkbox(True, label="Adresses FR")
            custom_entities = gr.Textbox(lines=3, label="Entités personnalisées (une par ligne, regex possible entre /.../)", placeholder="CCI Occitanie\n/\\bNomDeFamille\\b/")
            emit_original_text = gr.Checkbox(True, label="Écrire aussi le texte original (fichier séparé)")
            with gr.Row():
                ocr_enable = gr.Checkbox(True, label="Activer OCR auto")
                ocr_lang = gr.Textbox(value="fra+eng", label="Langues OCR (Tesseract)", scale=2)
                ocr_min_chars = gr.Number(value=15, precision=0, label="Min. chars/page pour éviter l'OCR", scale=1)
                ocr_force = gr.Checkbox(False, label="Forcer OCR sur toutes les pages")
            with gr.Row():
                save_prefs_btn = gr.Button("💾 Sauvegarder préférences")
                prefs_download = gr.File(label="Préférences (JSON)", interactive=False)
                load_prefs_file = gr.File(label="Charger préférences (JSON)")
                load_prefs_btn = gr.Button("↩️ Restaurer préférences")

        outdir_box = gr.Textbox(value="out", label="Dossier de sortie", placeholder="out")

        with gr.Row():
            run_btn = gr.Button("▶️ Exécuter")
            clear_btn = gr.Button("🧹 Nettoyer les sorties affichées")

        with gr.Tabs():
            with gr.Tab("Résultats Anonymisation"):
                redacted_pdf = gr.File(label="PDF redacted (diffusion)", interactive=False)
                anonymized_txt = gr.File(label="Texte anonymisé (réversible)", interactive=False)
                mapping_json = gr.File(label="Mapping (placeholders)", interactive=False)
                used_config = gr.File(label="Config utilisée", interactive=False)
                original_txt = gr.File(label="Texte original (optionnel)", interactive=False)
                preview_md = gr.Markdown()
            with gr.Tab("Résultats Désanonymisation"):
                deanonymized_txt = gr.File(label="Texte désanonymisé", interactive=False)
            with gr.Tab("Journal"):
                log_box = gr.Textbox(label="Logs", lines=18)

        # --- Fonctions helpers ---

        def collect_cfg(email, phone_fr, iban_fr, siren, siret, amount_eur, dates, person_names,
                        company_names, addresses_fr, custom_entities, emit_original_text,
                        ocr_enable, ocr_lang, ocr_min_chars, ocr_force):
            cfg = default_cfg_dict()
            cfg["categories"] = {
                "email": bool(email),
                "phone_fr": bool(phone_fr),
                "iban_fr": bool(iban_fr),
                "siren": bool(siren),
                "siret": bool(siret),
                "amount_eur": bool(amount_eur),
                "dates": bool(dates),
                "person_names": bool(person_names),
                "company_names": bool(company_names),
                "addresses_fr": bool(addresses_fr)
            }
            entities = []
            for line in (custom_entities or "").splitlines():
                line = line.strip()
                if line:
                    entities.append(line)
            cfg["custom_entities"] = entities
            cfg["output"]["emit_original_text"] = bool(emit_original_text)
            cfg["ocr"] = {
                "enable": bool(ocr_enable),
                "languages": ocr_lang or "fra+eng",
                "min_avg_chars_per_page": int(ocr_min_chars or 15),
                "force_ocr": bool(ocr_force)
            }
            return cfg

        def on_mode_change(m):
            if m == "Anonymiser":
                return (
                    gr.update(visible=True),  # input_pdf
                    gr.update(visible=False), # input_for_dean
                    gr.update(visible=False)  # mapping_in
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                )

        mode.change(on_mode_change, inputs=mode, outputs=[input_pdf, input_for_dean, mapping_in])

        def do_run(mode_val, input_pdf_f, input_for_dean_f, mapping_f,
                   email, phone_fr, iban_fr, siren, siret, amount_eur, dates, person_names,
                   company_names, addresses_fr, custom_entities, emit_original_text,
                   ocr_enable, ocr_lang, ocr_min_chars, ocr_force, outdir_text):
            # Reset outputs initially
            empty = [None, None, None, None, None, "", None, ""]
            try:
                outdir = Path(outdir_text or "out")
                if mode_val == "Anonymiser":
                    if not input_pdf_f:
                        return (*empty[:-1], "Veuillez fournir un PDF.",)
                    cfg = collect_cfg(email, phone_fr, iban_fr, siren, siret, amount_eur, dates, person_names,
                                      company_names, addresses_fr, custom_entities, emit_original_text,
                                      ocr_enable, ocr_lang, ocr_min_chars, ocr_force)
                    res = anonymize_pdf_gradio(input_pdf_f.name, cfg, outdir)
                    return (res["redacted_pdf"], res["anonymized_txt"], res["mapping_json"],
                            res["used_config"], (res["original_txt"] or None), res["preview"], None, res["log"])
                else:
                    if not input_for_dean_f or not mapping_f:
                        return (*empty[:-1], "Veuillez fournir le fichier anonymisé et le mapping.json.",)
                    res = deanonymize_file_gradio(input_for_dean_f.name, mapping_f.name, outdir)
                    return (None, None, None, None, None, "", res["deanonymized_txt"], res["log"])
            except Exception as e:
                return (*empty[:-1], f"Erreur: {e}")

        run_btn.click(
            do_run,
            inputs=[mode, input_pdf, input_for_dean, mapping_in,
                    email, phone_fr, iban_fr, siren, siret, amount_eur, dates, person_names,
                    company_names, addresses_fr, custom_entities, emit_original_text,
                    ocr_enable, ocr_lang, ocr_min_chars, ocr_force, outdir_box],
            outputs=[redacted_pdf, anonymized_txt, mapping_json, used_config, original_txt, preview_md, deanonymized_txt, log_box],
        )

        def clear_outputs():
            return (None, None, None, None, None, "", None, "")
        clear_btn.click(clear_outputs, inputs=None, outputs=[redacted_pdf, anonymized_txt, mapping_json, used_config, original_txt, preview_md, deanonymized_txt, log_box])

        # Sauvegarde / Restauration préférences
        def save_prefs(email, phone_fr, iban_fr, siren, siret, amount_eur, dates, person_names,
                       company_names, addresses_fr, custom_entities, emit_original_text,
                       ocr_enable, ocr_lang, ocr_min_chars, ocr_force):
            cfg = collect_cfg(email, phone_fr, iban_fr, siren, siret, amount_eur, dates, person_names,
                              company_names, addresses_fr, custom_entities, emit_original_text,
                              ocr_enable, ocr_lang, ocr_min_chars, ocr_force)
            tmp = Path(tempfile.mkdtemp(prefix="prefs_")) / "preferences.json"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            return str(tmp)

        save_prefs_btn.click(
            save_prefs,
            inputs=[email, phone_fr, iban_fr, siren, siret, amount_eur, dates, person_names,
                    company_names, addresses_fr, custom_entities, emit_original_text,
                    ocr_enable, ocr_lang, ocr_min_chars, ocr_force],
            outputs=prefs_download
        )

        def load_prefs(file_obj):
            if not file_obj:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            with open(file_obj.name, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg = AnonymizerConfig.from_dict(cfg)
            # Return values in UI order
            ce = "\n".join(cfg.custom_entities or [])
            return (cfg.categories.get("email", True),
                    cfg.categories.get("phone_fr", True),
                    cfg.categories.get("iban_fr", True),
                    cfg.categories.get("siren", True),
                    cfg.categories.get("siret", True),
                    cfg.categories.get("amount_eur", True),
                    cfg.categories.get("dates", True),
                    cfg.categories.get("person_names", True),
                    cfg.categories.get("company_names", True),
                    cfg.categories.get("addresses_fr", True),
                    ce,
                    cfg.output.get("emit_original_text", True),
                    cfg.ocr.get("enable", True),
                    cfg.ocr.get("languages", "fra+eng"),
                    int(cfg.ocr.get("min_avg_chars_per_page", 15)),
                    cfg.ocr.get("force_ocr", False))

        load_prefs_btn.click(
            load_prefs,
            inputs=[load_prefs_file],
            outputs=[email, phone_fr, iban_fr, siren, siret, amount_eur, dates, person_names,
                     company_names, addresses_fr, custom_entities, emit_original_text,
                     ocr_enable, ocr_lang, ocr_min_chars, ocr_force]
        )

    return demo

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo = build_ui()
    demo.queue(max_size=16).launch()

