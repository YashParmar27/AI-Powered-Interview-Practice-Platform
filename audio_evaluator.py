"""
audio_evaluator.py
------------------
Audio interview evaluation pipeline.
Pass in an audio file path and get back a voice score (1-5) + full breakdown.

Usage:
    python audio_evaluator.py path/to/interview.wav
    python audio_evaluator.py path/to/interview.wav --gender male

Dependencies:
    pip install librosa praat-parselmouth openai-whisper numpy scipy
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import librosa
import parselmouth
import whisper


# ===========================================================================
# CONFIG
# ===========================================================================

FRAME_DURATION_S       = 1.0    # 1-second frames throughout
N_MFCC                 = 13
WHISPER_MODEL          = "small"
PAUSE_THRESHOLD_S      = 0.3    # unfilled pause minimum
LONG_PAUSE_THRESHOLD_S = 1.0
BASELINE_PATH          = "baseline_mfcc.json"

PITCH_BANDS = {
    "male":    {"mean": (85,  180), "std": (25, 45)},
    "female":  {"mean": (165, 255), "std": (30, 55)},
    "unknown": {"mean": (85,  255), "std": (20, 55)},
}
MONOTONE_STD_THRESHOLD = 15.0

RMS_LOUDNESS_SCORES = [
    (45,   1),
    (55,   2),
    (60,   3),
    (70,   5),
    (80,   3),
    (None, 1),
]
RMS_CONSISTENCY_PENALTIES = [
    (3,    0.0),
    (6,    0.5),
    (10,   1.0),
    (None, 2.0),
]

MFCC_MEAN_THRESHOLDS = [10.0, 20.0, 35.0, 50.0]   # D1–D4, tune from calibration set
MFCC_VAR_THRESHOLDS  = [5.0,  12.0, 22.0, 35.0]   # V1–V4

WPM_SCORE_BANDS = [
    (90,   1),
    (120,  3),
    (160,  5),
    (190,  3),
    (None, 1),
]
FILLER_SCORE_BANDS = [
    (0.02, 5),
    (0.05, 4),
    (0.10, 3),
    (0.15, 2),
    (None, 1),
]
PAUSE_SCORE_BANDS = [
    (0.05, 5),
    (0.10, 4),
    (0.20, 3),
    (0.30, 2),
    (None, 1),
]

FILLER_WORDS = {"um", "uh", "er", "ah", "hmm"}
FILLER_PHRASES = [
    "you know", "i mean", "kind of", "sort of",
    "basically", "literally", "like", "right",
    "okay", "actually",
]

WEIGHTS = {
    "mfcc_clarity": 0.15,
    "pitch":        0.20,
    "rms_energy":   0.15,
    "speech_rate":  0.20,
    "filler_ratio": 0.15,
    "pauses":       0.15,
}


# ===========================================================================
# HELPERS
# ===========================================================================

def _clamp(v, lo=1.0, hi=5.0):
    return max(lo, min(hi, v))


def _band_score(value, bands):
    for ceiling, score in bands:
        if ceiling is None or value <= ceiling:
            return float(score)
    return 1.0


def _distance_to_score(distance, thresholds):
    for i, t in enumerate(thresholds):
        if distance <= t:
            return float(5 - i)
    return 1.0


def _loudness_score(rms_db):
    return _band_score(rms_db, RMS_LOUDNESS_SCORES)


def _consistency_penalty(rms_std_db):
    return _band_score(rms_std_db, RMS_CONSISTENCY_PENALTIES)


def _band_distance_score(val, lo, hi):
    """Score that drops symmetrically as val moves outside [lo, hi]."""
    if lo <= val <= hi:
        return 5.0
    midpoint   = (lo + hi) / 2
    half_width = (hi - lo) / 2
    deviation  = abs(val - midpoint) - half_width
    pct_off    = deviation / half_width
    if pct_off <= 0.20: return 4.0
    if pct_off <= 0.50: return 3.0
    if pct_off <= 0.80: return 2.0
    return 1.0


# ===========================================================================
# BRANCH A — ACOUSTIC FEATURES
# ===========================================================================

def _load_baseline():
    if not os.path.exists(BASELINE_PATH):
        return None, None
    with open(BASELINE_PATH) as f:
        data = json.load(f)
    return np.array(data["mean_vector"]), np.array(data["var_vector"])


def extract_mfcc(signal, sr):
    hop  = int(sr * FRAME_DURATION_S)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, hop_length=hop)

    speaker_mean = np.mean(mfcc, axis=1)
    speaker_var  = np.var(mfcc,  axis=1)

    baseline_mean, baseline_var = _load_baseline()

    if baseline_mean is None:
        
        return {
            "clarity_score":      3.0,
            "mean_distance":      -1.0,
            "variance_distance":  -1.0,
            "mean_score":         3.0,
            "variance_score":     3.0,
            "note": "No baseline found. Run build_baseline() to enable MFCC scoring."
        }

    dist_mean = float(np.linalg.norm(speaker_mean - baseline_mean))
    dist_var  = float(np.linalg.norm(speaker_var  - baseline_var))

    mean_score = _distance_to_score(dist_mean, MFCC_MEAN_THRESHOLDS)
    var_score  = _distance_to_score(dist_var,  MFCC_VAR_THRESHOLDS)
    clarity    = _clamp(0.6 * mean_score + 0.4 * var_score)

    return {
        "clarity_score":     round(clarity, 2),
        "mean_distance":     round(dist_mean, 2),
        "variance_distance": round(dist_var, 2),
        "mean_score":        round(mean_score, 2),
        "variance_score":    round(var_score, 2),
    }


def extract_pitch(signal, sr, gender="unknown"):
    snd       = parselmouth.Sound(signal, sampling_frequency=sr)
    pitch_obj = snd.to_pitch(time_step=FRAME_DURATION_S)
    f0_values = pitch_obj.selected_array["frequency"]
    voiced    = f0_values[f0_values > 0]

    if len(voiced) == 0:
        return {
            "pitch_score": 1.0, "f0_mean": 0.0, "f0_std": 0.0,
            "pct_voiced": 0.0,  "monotone": True
        }

    f0_mean    = float(np.mean(voiced))
    f0_std     = float(np.std(voiced))
    pct_voiced = float(len(voiced) / len(f0_values))

    if f0_std < MONOTONE_STD_THRESHOLD:
        return {
            "pitch_score": 1.0, "f0_mean": round(f0_mean, 1),
            "f0_std": round(f0_std, 1), "pct_voiced": round(pct_voiced, 3),
            "monotone": True
        }

    band       = PITCH_BANDS.get(gender, PITCH_BANDS["unknown"])
    m_lo, m_hi = band["mean"]
    s_lo, s_hi = band["std"]

    mean_score  = _band_distance_score(f0_mean, m_lo, m_hi)
    std_score   = _band_distance_score(f0_std,  s_lo, s_hi)
    pitch_score = _clamp(0.5 * mean_score + 0.5 * std_score)

    return {
        "pitch_score": round(pitch_score, 2),
        "f0_mean":     round(f0_mean, 1),
        "f0_std":      round(f0_std, 1),
        "pct_voiced":  round(pct_voiced, 3),
        "monotone":    False,
    }


def extract_rms(signal, sr):
    hop          = int(sr * FRAME_DURATION_S)
    rms          = librosa.feature.rms(y=signal, hop_length=hop)[0]
    rms          = np.where(rms == 0, 1e-10, rms)
    rms_db_series = 20 * np.log10(rms)

    rms_db     = float(np.mean(rms_db_series))
    rms_std_db = float(np.std(rms_db_series))

    loud_score   = _loudness_score(rms_db)
    penalty      = _consistency_penalty(rms_std_db)
    energy_score = _clamp(loud_score - penalty)

    return {
        "energy_score":        round(energy_score, 2),
        "rms_db":              round(rms_db, 1),
        "rms_std_db":          round(rms_std_db, 1),
        "loudness_score":      round(loud_score, 2),
        "consistency_penalty": round(penalty, 2),
    }


# ===========================================================================
# BRANCH B — LINGUISTIC FEATURES
# ===========================================================================

_whisper_model_cache = None

def _get_whisper():
    global _whisper_model_cache
    if _whisper_model_cache is None:
        print(f"[whisper] Loading '{WHISPER_MODEL}' model...")
        _whisper_model_cache = whisper.load_model(WHISPER_MODEL)
    return _whisper_model_cache


def _flatten_words(result):
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word":  w["word"].strip().lower(),
                "start": w["start"],
                "end":   w["end"],
            })
    return words


def _count_filler_phrases(transcript):
    count = 0
    text  = transcript.lower()
    for phrase in FILLER_PHRASES:
        count += len(re.findall(r'\b' + re.escape(phrase) + r'\b', text))
    return count


def compute_speech_rate(words, total_duration, filler_count):
    total_words = len(words)
    non_filler  = max(total_words - filler_count, 0)

    total_pause_time = 0.0
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap > PAUSE_THRESHOLD_S:
            total_pause_time += gap

    speech_time = max(total_duration - total_pause_time, 1.0)
    wpm         = (non_filler / speech_time) * 60.0

    return {
        "wpm":                round(wpm, 1),
        "speech_rate_score":  _band_score(wpm, WPM_SCORE_BANDS),
        "total_words":        total_words,
        "speech_time_s":      round(speech_time, 2),
    }


def compute_fillers(words, transcript):
    total_words         = len(words)
    single_filler_count = sum(1 for w in words if w["word"] in FILLER_WORDS)
    phrase_filler_count = _count_filler_phrases(transcript)
    total_fillers       = single_filler_count + phrase_filler_count
    filler_ratio        = total_fillers / max(total_words, 1)

    return {
        "filler_ratio": round(filler_ratio, 4),
        "filler_score": _band_score(filler_ratio, FILLER_SCORE_BANDS),
        "filler_count": total_fillers,
        "total_words":  total_words,
    }


def compute_pauses(words):
    if len(words) < 2:
        return {
            "pause_score": 5.0, "pause_rate": 0.0, "pause_count": 0,
            "long_pause_count": 0, "mean_long_pause_s": 0.0
        }

    pauses, long_pauses = [], []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap > PAUSE_THRESHOLD_S:
            pauses.append(gap)
        if gap > LONG_PAUSE_THRESHOLD_S:
            long_pauses.append(gap)

    pause_rate = len(pauses) / max(len(words), 1)
    mean_long  = float(np.mean(long_pauses)) if long_pauses else 0.0

    return {
        "pause_score":        _band_score(pause_rate, PAUSE_SCORE_BANDS),
        "pause_rate":         round(pause_rate, 4),
        "pause_count":        len(pauses),
        "long_pause_count":   len(long_pauses),
        "mean_long_pause_s":  round(mean_long, 2),
    }


def run_linguistic_analysis(audio_path):
    model      = _get_whisper()
    result     = model.transcribe(audio_path, word_timestamps=True, language="en")
    transcript = result.get("text", "").strip()
    words      = _flatten_words(result)

    if words:
        total_duration = words[-1]["end"]
    else:
        segs           = result.get("segments", [])
        total_duration = segs[-1]["end"] if segs else 1.0

    filler_result = compute_fillers(words, transcript)

    return {
        "speech_rate": compute_speech_rate(words, total_duration,
                                           filler_result["filler_count"]),
        "fillers":     filler_result,
        "pauses":      compute_pauses(words),
        "transcript":  transcript,
    }


# ===========================================================================
# AGGREGATION
# ===========================================================================

def compute_final_score(acoustic, linguistic):
    scores = {
        "mfcc_clarity": acoustic["mfcc"]["clarity_score"],
        "pitch":        acoustic["pitch"]["pitch_score"],
        "rms_energy":   acoustic["rms"]["energy_score"],
        "speech_rate":  linguistic["speech_rate"]["speech_rate_score"],
        "filler_ratio": linguistic["fillers"]["filler_score"],
        "pauses":       linguistic["pauses"]["pause_score"],
    }

    voice_score = round(_clamp(sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)), 2)

    flags = {
        "monotone_delivery":    acoustic["pitch"]["monotone"],
        "excessive_fillers":    linguistic["fillers"]["filler_ratio"] > 0.10,
        "too_fast":             linguistic["speech_rate"]["wpm"] > 190,
        "too_slow":             linguistic["speech_rate"]["wpm"] < 90,
        "frequent_long_pauses": linguistic["pauses"]["long_pause_count"] >= 3,
        "low_energy":           acoustic["rms"]["rms_db"] < 50,
        "inconsistent_volume":  acoustic["rms"]["rms_std_db"] > 10,
    }

    return {
        "voice_score": voice_score,
        "sub_scores":  {k: round(v, 2) for k, v in scores.items()},
        "breakdown": {
            "acoustic":   acoustic,
            "linguistic": linguistic,
        },
        "flags":        flags,
        "weights_used": WEIGHTS,
    }


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def evaluate(audio_path, gender="unknown"):
    """
    Full pipeline: audio file → voice score dict.

    Parameters
    ----------
    audio_path : str    Path to .wav / .mp3 / .m4a
    gender     : str    "male" | "female" | "unknown"

    Returns
    -------
    dict with voice_score, sub_scores, breakdown, flags
    """
    print(f"[pipeline] Loading audio: {audio_path}")
    signal, sr = librosa.load(audio_path, sr=None, mono=True)

    print("[pipeline] Running acoustic analysis...")
    acoustic = {
        "mfcc":  extract_mfcc(signal, sr),
        "pitch": extract_pitch(signal, sr, gender=gender),
        "rms":   extract_rms(signal, sr),
    }

    print("[pipeline] Running linguistic analysis...")
    linguistic = run_linguistic_analysis(audio_path)

    print("[pipeline] Computing final score...")
    output = compute_final_score(acoustic, linguistic)

    return output


# ===========================================================================
# OPTIONAL: BASELINE BUILDER
# ===========================================================================

def build_baseline(calibration_audio_paths, save_path=BASELINE_PATH):
    """
    Build and save the MFCC baseline from fluent-speaker calibration clips.
    Call once before using MFCC scoring.

    Parameters
    ----------
    calibration_audio_paths : list of str
    save_path               : str
    """
    all_means, all_vars = [], []
    for path in calibration_audio_paths:
        signal, sr = librosa.load(path, sr=None, mono=True)
        hop        = int(sr * FRAME_DURATION_S)
        mfcc       = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, hop_length=hop)
        all_means.append(np.mean(mfcc, axis=1).tolist())
        all_vars.append(np.var(mfcc, axis=1).tolist())

    baseline = {
        "mean_vector": np.mean(all_means, axis=0).tolist(),
        "var_vector":  np.mean(all_vars,  axis=0).tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"[baseline] Saved to {save_path} ({len(calibration_audio_paths)} clips used)")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate interview audio clip.")
    parser.add_argument("audio_path", help="Path to audio file (.wav, .mp3, .m4a)")
    parser.add_argument("--gender", default="unknown",
                        choices=["male", "female", "unknown"],
                        help="Speaker gender for pitch scoring")
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty-print the JSON output")
    args = parser.parse_args()

    result = evaluate(args.audio_path, gender=args.gender)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

indent = 2 if args.pretty else None
print(json.dumps(result, indent=indent, cls=NumpyEncoder))