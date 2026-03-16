"""
ADD THIS ENDPOINT to routes_ai.py

Paste the code below at the end of routes_ai.py, after the existing
classify_ecg_endpoint and mix_stems endpoints.

It also requires this import at the top of routes_ai.py
(add alongside the existing ecg_wrapper imports):
    from ai.ecg_wrapper import ecg_ica_separate, _ICA_AVAILABLE, classify_ecg, classify_ecg_full
"""

# ─── New endpoint — add to the BOTTOM of routes_ai.py ────────────────────────

@router.post("/classify_ecg_full")
def classify_ecg_full_endpoint(req: AIProcessRequest):
    """
    Full 12-channel ECG classification with slider gain support.

    Loads {file_id}_12ch.csv (saved by the upload route when a CSV is
    uploaded), applies per-disease bandpass gains, runs the Keras ResNet,
    and returns:
      - diagnosis (is_diseased, scores, detected/suspected diseases)
      - 12-channel lead data for the ECGDiagnosis viewer
      - highlighted lead indices (most diagnostic leads per disease)

    Body: { file_id, mode: "ecg", gains: [float * 7] }
      gains order: [Normal, 1dAVb, RBBB, LBBB, SB, AF, ST]
      gain = 0.0 → suppress that disease's spectral signature → score drops
      gain = 2.0 → amplify it → score rises

    Falls back to 1D tiling when no _12ch.csv is found (non-CSV uploads).
    """
    from ai.ecg_wrapper import classify_ecg_full

    result = classify_ecg_full(
        file_id    = req.file_id,
        gains      = req.gains if req.gains else [],
        upload_dir = UPLOAD_DIR,
    )
    logger.info(
        "ECG full classification endpoint",
        extra={
            "file_id":    req.file_id,
            "detected":   result.get("detected_diseases", []),
            "suspected":  result.get("suspected_diseases", []),
            "is_diseased": result["is_diseased"],
            "gains_provided": bool(req.gains),
        }
    )
    return result
