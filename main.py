#!/usr/bin/env python3
import re
import json, argparse, datetime

import json, argparse
from pathlib import Path
import requests

from sheet_upsert import init_upsert_process


# -------- Transcriber (faster-whisper) --------
def load_whisper(model_path_or_name="large-v3", device="auto", compute_type="float16"):
    from faster_whisper import WhisperModel
    return WhisperModel(model_path_or_name, device=device, compute_type=compute_type)

def transcribe_audio(model, audio_path, force_lang="id"):
    """
    Force Bahasa Indonesia ('id') for best accuracy, then the summariser will output English.
    """
    segments, info = model.transcribe(
        str(audio_path),
        language=force_lang,           # <-- Bahasa input
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        beam_size=5, temperature=0
    )
    return " ".join(s.text.strip() for s in segments).strip()

# -------- Summariser (Ollama, local) --------
def ollama_chat_json(transcript, meta, model="qwen3:8b", host="http://localhost:11434"):
    """
    Insight-focused summariser for your 10-column sheet schema.
    - Input may be Indonesian; output is EN (British).
    - Synthesises impressions across the whole recording (not just cherry-picks).
    """
    import json, datetime, requests

    system_prompt = """You summarise Margherita pizza voice notes to a CONSISTENT schema.
OUTPUT MUST be clear ENGLISH (British English).
Be conservative: if the transcript doesn't say it, use "" (empty string).
Use concrete, food-critic descriptors; avoid flowery language.
Tier: S (outstanding), A (excellent), B (good), C (average), D (below average), E (poor), F (horrible).

INSIGHT OVER COVERAGE:
- Read the ENTIRE transcript and capture how impressions EVOLVE over time (e.g., first bite vs mid-slice vs after cooling).
- Prefer synthesised takeaways over isolated facts.
- If trade-offs are mentioned (e.g., light sauce but clean finish; airy rim but underbaked centre), make them explicit.
- If price/queue/service context appears, weave it briefly into Overall (value for money, practicality).

FIELD GUIDANCE (keep each field concise):
- Location: name of pizzeria + nearby landmark/area if stated (e.g., “Suditalia — near Borough Market”).
- Crust: dough/structure/bake/handling in 2–3 phrases (e.g., “airy cornicione, leopard-spotting, slight flop”).
- Sauce: ripeness/acidity/sweetness/salt/clarity in 1–2 phrases.
- Cheese: melt, salinity, oiliness in 1–2 phrases.
- Basil/Extras: if basil/daun basil/kemangi/selasih is mentioned, note freshness/aroma (“fresh, aromatic”) or “mentioned”; else "".
- Balance/Harmony: how elements worked together ACROSS the whole slice; note changes over time if mentioned.
- Appearance/Aroma: visual + aroma (e.g., “leopard-spotting; gentle wood-char aroma”).
- Overall: 1–2 sentences that synthesise your verdict, value-for-money/queue/service if present, and who would like it.
- Tier: S/A/B/C/D/E/F only. If numeric scores are mentioned (e.g., 8-6-10), convert to a single tier conservatively (lean lower on mixed signals).

TONE:
- If the pizza is weak, use precise, non-condescending phrasing (e.g., “underseasoned sauce; soft centre”) rather than harsh language.
Return ONLY one JSON object with the exact keys below. No extra text, no code fences.
"""

    # NOTE: Keep exactly these 10 keys to match your Google Sheet schema.
    user_prompt = f"""
Return ONLY valid JSON with these exact keys (all strings):
{{
  "Date": "",
  "Location": "",
  "Crust": "",
  "Sauce": "",
  "Cheese": "",
  "Basil/Extras": "",
  "Balance/Harmony": "",
  "Appearance/Aroma": "",
  "Overall": "",
  "Tier": ""
}}

Rules:
- Populate ONLY from the transcript (and META if provided).
- Keep fields tight and information-dense; avoid generic adjectives.
- OUTPUT LANGUAGE: ENGLISH ONLY.
- If landmarks/area are in the transcript or META, include them in Location (e.g., “— near Trafalgar Square”).

TRANSCRIPT (may be Indonesian):
{transcript}

META (may be partial JSON):
{json.dumps(meta, ensure_ascii=False)}
""".strip()

    payload = {
        "model": model,
        "options": {"temperature": 0.1, "repeat_penalty": 1.1, "num_ctx": 8192},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False
    }
    r = requests.post(f"{host}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    content = r.json().get("message", {}).get("content", "")

    # Robust JSON extraction
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output:\n" + content)
    js = json.loads(content[start:end+1])

    # Auto-fill Date if blank (ISO yyyy-mm-dd)
    if not js.get("Date"):
        js["Date"] = datetime.datetime.now().strftime("%Y-%m-%d")
    return js


# -------- I/O helpers --------
def load_sidecar_meta(audio_path: Path):
    """
    Optional sidecar: note.m4a.meta.json
    Example: {"pizzeria":"50 Kalò London","neighbourhood":"Trafalgar Square","price_gbp":"11","portion":"Whole"}
    """
    sidecar = Path(str(audio_path) + ".meta.json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text())
        except Exception:
            pass
    return {}

def save_outputs(outdir: Path, stem: str, transcript: str, summary: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{stem}.transcript-id.txt").write_text(transcript)   # Indonesian transcript
    (outdir / f"{stem}.summary-en.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

DEFAULT_ORDER = [
    "date","time","pizzeria","neighbourhood","address","lat","lon","price_gbp",
    "portion","oven_type","crust_notes","sauce_notes","cheese_notes","basil_toppings_notes",
    "balance_notes","appearance_aroma_notes","overall_impression","tier","would_return",
    "photo_url","audio_url","raw_transcript","tags","wait_time_min"
]

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Local Bahasa→English pipeline: faster-whisper (ID) + Ollama (EN summary).")
    ap.add_argument("inputs", nargs="+", help="Audio files (.m4a/.mp3/.wav)")
    ap.add_argument("--whisper", default="large-v3", help="Whisper model or path")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/metal")
    ap.add_argument("--compute-type", default="int8", help="int8/float16/float32/auto")
    ap.add_argument("--ollama-model", default="qwen2.5:7b-instruct", help="e.g., qwen3:8b")
    ap.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host")
    ap.add_argument("--outdir", default="./out", help="Directory for outputs")
    ap.add_argument("--print", action="store_true", help="Also print JSON to stdout")
    args = ap.parse_args()

    wh = load_whisper(args.whisper, device=args.device, compute_type=args.compute_type)

    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            print(f"[skip] not found: {p}")
            continue

        print(f"→ Transcribing (Bahasa → text): {p.name}")
        transcript_id = transcribe_audio(wh, p, force_lang="id")

        meta = load_sidecar_meta(p)

        print(f"→ Summarising to EN with {args.ollama_model}")
        summary_en = ollama_chat_json(
            transcript=transcript_id,
            meta=meta,
            model=args.ollama_model,
            host=args.ollama_host
        )

        save_outputs(Path(args.outdir), p.stem, transcript_id, summary_en)

        if args.print:
            print(json.dumps(summary_en, ensure_ascii=False, indent=2))

        print(f"✓ Done: {p.stem}.summary-en.json")

def rm_all_files(path="/Users/setra.wicana/Downloads/pizza_summariser/out"):
    return sum((p.unlink() or 1) for p in Path(path).iterdir() if p.is_file())

if __name__ == "__main__":
    rm_all_files()
    # ollama pull qwen2.5:7b-instruct
    # python main.py "/Users/setra.wicana/Documents/recording/test-suara.m4a" --ollama-model qwen2.5:7b-instruct
    main()
    json_transcribed_path = '/Users/setra.wicana/Downloads/pizza_summariser/out/'
    sheet_id = '1frNDLv0WCm51Yz0iNlIp-4LXnddvdJJ8Rso70jCKPg4'
    sheet_tab = 'Sheet1'
    service_acccount_path = '/Users/setra.wicana/Downloads/pizza_summariser/pizza-project-475415-32b2e203158a.json'
    print('start upserting process to Google Sheet...')

    init_upsert_process(json_transcribed_path, sheet_id, sheet_tab, service_acccount_path)
