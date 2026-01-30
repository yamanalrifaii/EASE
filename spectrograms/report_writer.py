# report_writer.py
import os, json, base64, requests

GEMINI_API_KEY = ""
# Use the same model you used in your training script
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def draft_report(spectrogram_png, gradcam_png, cnn_json):
    """
    spectrogram_png: path to spectrogram image
    gradcam_png: path to Grad-CAM overlay image
    cnn_json: dict containing model findings
    Returns: dict with structured clinical-style summary
    """
    prompt = f"""
You are generating a clinician-style explanation for an EEG spectrogram seizure detector.
You are given:
1. Model findings (CNN JSON)
2. Two images: the original spectrogram and a Grad-CAM overlay showing areas of model attention.

Use BOTH to write an explainable clinical summary.

Rules:
- Begin with a concise paragraph summarizing whether seizure activity is present and the model's confidence.
- Mention Grad-CAM findings (where the model focused: time/frequency zones, % highlighted).
- List seizure events (start, end, confidence).
- Comment on artifacts/quality if obvious.
- Give 2–3 clinician recommendations.
- Always include a clear disclaimer that the report is AI-generated and for assistive use only.

Return STRICT JSON with keys:
{{
  "summary": str,
  "xai_explanation": str,
  "events": [{{"start_sec": float, "end_sec": float, "confidence": float}}],
  "artifacts": [str],
  "recommendations": [str],
  "disclaimer": str
}}
CNN JSON:
{json.dumps(cnn_json, indent=2)}
"""


    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": _b64(spectrogram_png)}},
                {"inline_data": {"mime_type": "image/png", "data": _b64(gradcam_png)}}
            ]
        }],
        "generationConfig": {"temperature": 0.2}
    }

    r = requests.post(url, json=body, timeout=90)
    r.raise_for_status()
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    s, e = txt.find("{"), txt.rfind("}")
    return json.loads(txt[s:e+1])
