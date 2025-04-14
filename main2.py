#!/usr/bin/env python3
"""
  Production-Ready Soundscape Generator with:
    1) Visual Timeline Plot
    2) Reliable 'dog_bark' and 'paper_rustle' triggers
    3) Markov-based 'fear stabs' (replaces old tone layering)
    4) Rule-based crescendo for repeated fear references

USAGE:
  python main2.py --input story.txt --output soundscape.wav
  Optional:
    --wpm 150
    --model sound_effect_model.pkl
    (We removed the old mood/pacing override logic for clarity)

Make sure you have first trained your model with train.py to generate 'sound_effect_model.pkl'.
"""

import argparse
import logging
import os
import random
import sys
import pickle
import re
from typing import List, Dict, Tuple

import nltk
import spacy
import numpy as np
from nltk.corpus import wordnet as wn, stopwords
from textblob import TextBlob
from pydub import AudioSegment, exceptions
from pydub.generators import Sine
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##################################################
#             LOAD SPACy & CHECK NLTK
##################################################
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("SpaCy model 'en_core_web_sm' not found. Install with:\n  python -m spacy download en_core_web_sm")
    sys.exit(1)

# In case user hasn't downloaded stopwords or wordnet data:
try:
    wn.synsets("dog")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

##################################################
#                   CONFIG
##################################################

class Config:
    AUDIO_DIR = "audio"              # Directory for audio files
    OUTPUT_FORMAT = "wav"
    BASE_LOOP_DURATION = 30_000      # 30 seconds minimum
    SENTIMENT_THRESHOLDS = {"positive": 0.15, "negative": -0.15}

    # Volume offsets (dB)
    VOLUME_EFFECTS = -12

    # If you want to reduce the repeated effect spam further
    EFFECT_COOLDOWN_MS = 4000

    # Maximum # of effects per sentence
    MAX_EFFECTS_PER_SENTENCE = 3

    # Fear crescendo config
    FEAR_THRESHOLD = 2       # # of times we detect "fear" in last N tokens
    FEAR_MEMORY = 5          # lookback in tokens
    FEAR_VOL_INCREMENT = 4   # each time we exceed threshold, increment heartbeat volume (dB)
    FEAR_MAX_BOOST = 10      # never boost more than +10 dB

##################################################
#         LABEL → AUDIO MAPPINGS
##################################################

# Must have these audio files in your "audio/" folder, or the code
# will fail back to silence for missing ones.

LABEL_TO_EFFECTS = {
    "storm":            ["effect_thunder.wav", "effect_rain.wav"],
    "rain":             ["effect_rain.wav"],
    "wind":             ["effect_wind.wav"],
    "night_silence":    ["effect_ominous_silence.wav"],
    "footsteps_wood":   ["effect_footsteps_wood.wav"],
    "footsteps_stone":  ["effect_footsteps_stone.wav"],
    "old_mansion":      ["effect_creaking_floor.wav"],
    "running":          ["effect_fast_footsteps.wav"],
    "train":            ["effect_train.wav"],
    "paper_rustle":     ["effect_paper.wav"],
    "heartbeat":        ["effect_heartbeat.wav"],
    "dog_bark":         ["effect_dog_barking.wav"],
    "generic_effect":   ["effect_generic.wav"]
}

# If synonyms or fallback trigger these keys,
# we map them to the above effect arrays
SYNONYMS = {
    "ocean":   ["sea", "waves", "water"],
    "volcano": ["eruption", "lava", "ash"],
    "forest":  ["woods", "trees"],
    "thunder": ["lightning", "storm"],
    "rain":    ["shower", "drizzle", "downpour"],
    "wind":    ["breeze", "gust", "draft"],
    # For sure dog references:
    "dog_bark": ["dog", "dogs", "cart", "dogcart"],
    # Paper references:
    "paper_rustle": ["paper", "book", "desk", "casebook", "document", "letter"]
}

AUDIO_CACHE = {}
MODEL_PIPELINE = None  # The loaded ML model pipeline

##################################################
#  ML-BASED PREDICTION
##################################################

def load_model(model_path: str) -> None:
    global MODEL_PIPELINE
    if MODEL_PIPELINE is not None:
        return  # already loaded
    try:
        with open(model_path, "rb") as f:
            MODEL_PIPELINE = pickle.load(f)
        logger.info(f"✅ Loaded ML model from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}. No ML predictions.")
        MODEL_PIPELINE = None

def predict_effect_label(token_text: str) -> str:
    """
    Return the predicted effect label or 'generic_effect' if the model is not confident.
    Lower threshold to allow more variety.
    """
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        return "generic_effect"

    probas = MODEL_PIPELINE.predict_proba([token_text])[0]
    label_idx = probas.argmax()
    confidence = probas[label_idx]
    predicted_label = MODEL_PIPELINE.classes_[label_idx]
    # Example threshold of 0.20
    if confidence < 0.20:
        return "generic_effect"
    return predicted_label

##################################################
#  FALLBACK: Synonyms + WordNet
##################################################

from nltk.corpus import wordnet as wn

def find_synonym_match(word: str) -> str:
    # Direct synonyms dictionary
    for label, syns in SYNONYMS.items():
        if word in syns:
            return label

    # WordNet fallback
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace("_", "").lower()
            if lemma_name in LABEL_TO_EFFECTS:
                return lemma_name

    return None

def fallback_effect_label(token_text: str) -> str:
    matched = find_synonym_match(token_text)
    if matched and matched in LABEL_TO_EFFECTS:
        return matched
    return "generic_effect"

##################################################
#  get_effects_for_token
##################################################

def get_effects_for_token(token_text: str) -> List[str]:
    """
    1) Attempt ML label.
    2) If unknown, fallback synonyms.
    """
    token_text = token_text.lower().strip()
    # Hard-coded overrides:
    if token_text in ["dog", "cart", "dogcart"]:
        return ["effect_dog_barking.wav"]
    if token_text in ["paper", "book", "desk", "document", "casebook", "letter"]:
        return ["effect_paper.wav"]

    # ML path
    ml_label = predict_effect_label(token_text)
    if ml_label in LABEL_TO_EFFECTS:
        return LABEL_TO_EFFECTS[ml_label]

    # fallback synonyms
    fallback_label = fallback_effect_label(token_text)
    if fallback_label in LABEL_TO_EFFECTS:
        return LABEL_TO_EFFECTS[fallback_label]

    # if all else fails
    return LABEL_TO_EFFECTS["generic_effect"]

##################################################
#  AUDIO UTILS
##################################################

from pydub import AudioSegment, exceptions

def load_audio(filename: str) -> AudioSegment:
    if filename in AUDIO_CACHE:
        return AUDIO_CACHE[filename]

    path = os.path.join(Config.AUDIO_DIR, filename)
    if not os.path.exists(path):
        logger.warning(f"Missing audio: {filename}")
        AUDIO_CACHE[filename] = AudioSegment.silent(duration=2000)
        return AUDIO_CACHE[filename]

    try:
        seg = AudioSegment.from_file(path)
        AUDIO_CACHE[filename] = seg
    except exceptions.CouldntDecodeError:
        logger.error(f"Failed to decode {filename}, using silence.")
        seg = AudioSegment.silent(duration=2000)
        AUDIO_CACHE[filename] = seg

    return seg

##################################################
#  TEXT ANALYSIS
##################################################

def analyze_text(text: str) -> Tuple[str, float]:
    """
    Return overall sentiment label and polarity for reference
    """
    polarity = TextBlob(text).sentiment.polarity
    if polarity > Config.SENTIMENT_THRESHOLDS["positive"]:
        mood = "positive"
    elif polarity < Config.SENTIMENT_THRESHOLDS["negative"]:
        mood = "negative"
    else:
        mood = "neutral"
    return mood, polarity

def split_text_into_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]

def estimate_sentence_durations(text: str, wpm=150) -> List[int]:
    sents = split_text_into_sentences(text)
    durations = []
    words_per_sec = wpm / 60.0
    for sent in sents:
        wc = len(sent.split())
        read_time_s = wc / words_per_sec
        durations.append(int(read_time_s * 1000))
    return durations

##################################################
#  STEP: RULE-BASED MARKOV + FEAR STABS
##################################################

def generate_fear_stabs(length_ms=30000, chord=None) -> AudioSegment:
    """
    A simple Markov-like chord stab approach for the 'fear' mood,
    or just for adding tension. We overlay random stabs if fear is high.
    """
    if chord is None:
        chord = 220.0  # A3 for tension

    seg = AudioSegment.silent(duration=length_ms)
    # Each stab is ~1 sec
    ms_per_stab = 1000
    fade_ms = 100
    t = 0
    while t < length_ms:
        # 20% chance of playing a stab
        if random.random() < 0.20:
            sine = Sine(chord).to_audio_segment(duration=ms_per_stab)
            sine = sine.fade_in(fade_ms).fade_out(fade_ms).apply_gain(-18)  # quiet stabs
            seg = seg.overlay(sine, position=t)
        t += ms_per_stab
    return seg

##################################################
#  MAIN SOUND PLACEMENT (with fear crescendo)
##################################################

def place_effects_with_timing(base_track: AudioSegment,
                              text: str,
                              durations: List[int],
                              log_csv: str = "train/soundscape_timeline.csv",
                              log_txt: str = "train/soundscape_log.txt") -> AudioSegment:
    """
    Places token-level effects, logs timeline to CSV for plotting,
    and also logs a high-level sentence-based summary.

    We'll do a 'fear memory' approach:
      - keep track of last N tokens for 'fear'
      - if count of 'fear' >= FEAR_THRESHOLD, boost heartbeat volume
    """
    import time

    final = base_track
    current_pos = 0
    sents = split_text_into_sentences(text)

    # We'll keep a CSV timeline with rows: timestamp_ms,effect,sentence_idx,token
    # so we can visualize later
    timeline_lines = ["timestamp_ms,effect,sentence_idx,token\n"]

    # We'll also keep a log of the final applied per sentence
    sentence_log = []

    # For cooldown and fear memory
    last_applied = {}
    fear_history = []
    heartbeat_boost = 0

    for i, sent in enumerate(sents):
        dur = durations[i]
        doc = nlp(sent)
        tokens = list(doc)
        n_tokens = len(tokens)

        applied_effects = set()
        applied_this_sentence = 0

        for j, tok in enumerate(tokens):
            word_raw = tok.text.lower().strip()
            word = re.sub(r"[^a-z0-9\s]", "", word_raw)

            # Update fear memory
            if "fear" in word:
                fear_history.append(time.time())  # store a time or just store the index

            # Trim fear_history to the last N tokens or last few seconds
            if len(fear_history) > Config.FEAR_MEMORY:
                fear_history = fear_history[-Config.FEAR_MEMORY:]

            # If fear references exceed threshold, boost heartbeat
            if len(fear_history) >= Config.FEAR_THRESHOLD:
                # apply limited boost
                heartbeat_boost = min(Config.FEAR_MAX_BOOST, heartbeat_boost + Config.FEAR_VOL_INCREMENT)
                # clear it to require repeated triggers
                fear_history.clear()

            fx_files = get_effects_for_token(word)
            if not fx_files or fx_files == ["effect_generic.wav"]:
                continue

            # Limit effect spam
            if applied_this_sentence >= Config.MAX_EFFECTS_PER_SENTENCE:
                break

            fraction = j / max(1, n_tokens - 1)
            pos = current_pos + int(dur * fraction)

            for fx_file in fx_files:
                # check cooldown
                if fx_file in last_applied:
                    if (pos - last_applied[fx_file]) < Config.EFFECT_COOLDOWN_MS:
                        continue

                fx_audio = load_audio(fx_file)

                # If effect is heartbeat, apply fear-based volume boost
                # limited by +10 dB in config
                effective_gain = Config.VOLUME_EFFECTS
                if "heartbeat" in fx_file and heartbeat_boost > 0:
                    effective_gain = Config.VOLUME_EFFECTS + heartbeat_boost

                # random +/- 2 dB
                effective_gain += random.randint(-2, 2)

                fx_audio = fx_audio.fade_in(100).fade_out(100).apply_gain(effective_gain)

                final = final.overlay(fx_audio, position=pos)
                last_applied[fx_file] = pos
                applied_effects.add(fx_file)
                applied_this_sentence += 1

                timeline_lines.append(f"{pos},{fx_file},{i+1},{word}\n")

        # done with sentence
        sentence_log.append(f"Section {i+1}: \"{sent}\" → Applied {sorted(list(applied_effects))}")
        current_pos += dur

    # Write timeline CSV
    with open(log_csv, "w", encoding="utf-8") as f:
        f.writelines(timeline_lines)
    logger.info(f"Timeline CSV written to: {log_csv}")

    # Write sentence-level log
    with open(log_txt, "w", encoding="utf-8") as f:
        for line in sentence_log:
            f.write(line + "\n")
    logger.info(f"Sentence-level log written to: {log_txt}")

    return final

##################################################
#  VISUALIZE THE TIMELINE
##################################################

def visualize_soundscape(csv_path="train/soundscape_timeline.csv", out_fig="train/soundscape_timeline.png"):
    """
    Reads the timeline CSV produced by place_effects_with_timing
    and plots a horizontal timeline chart of effect overlays.
    Saves to out_fig (e.g. 'soundscape_timeline.png').
    """
    import csv
    import matplotlib.pyplot as plt

    # We'll parse each row: timestamp_ms,effect,sentence_idx,token
    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = int(row["timestamp_ms"])
            effect = row["effect"]
            section = int(row["sentence_idx"])
            token = row["token"]
            events.append((timestamp, effect, section, token))

    if not events:
        logger.warning("No timeline events found. Check CSV content.")
        return

    # Sort by time
    events.sort(key=lambda x: x[0])

    # We'll assign each effect a numeric 'track' so we can visualize stacked lines
    effect_tracks = {}
    next_track_id = 0

    # We'll store: (timestamp, track_id, effect, token)
    plotted = []
    for (ts, fx, sect, tok) in events:
        if fx not in effect_tracks:
            effect_tracks[fx] = next_track_id
            next_track_id += 1
        track_id = effect_tracks[fx]
        plotted.append((ts, track_id, fx, tok))

    # Now let's plot them
    plt.figure(figsize=(12, 6))
    for (ts, track_id, fx, tok) in plotted:
        # We'll represent each effect event as a short bar at (ts, track_id)
        # length of bar = 1000ms just for visualization
        bar_length = 800  # ms
        plt.plot([ts, ts+bar_length], [track_id, track_id], lw=6)

        # optional: we can annotate the token or effect
        # plt.text(ts, track_id + 0.1, fx, fontsize=8)

    # Setup y-axis
    yticks = []
    ylabels = []
    for effect, track in effect_tracks.items():
        yticks.append(track)
        ylabels.append(effect)
    plt.yticks(yticks, ylabels)

    plt.xlabel("Time (ms)")
    plt.title("Soundscape Timeline")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(out_fig, dpi=150)
    logger.info(f"Soundscape timeline chart saved to: {out_fig}")
    plt.close()

##################################################
#   MAIN PIPELINE
##################################################

def create_soundscape_full(
    text: str,
    output_path: str = "soundscape_out.wav",
    wpm: int = 150,
    model_path: str = None
) -> None:
    """
    1) Load model
    2) Analyze text just for reference
    3) Create a base silent track of at least 30s
    4) Place token-level effects (with fear crescendo)
    5) Then overlay markov-based fear stabs if the text is negative or has 'fear'
    6) Output final
    7) Also produce a timeline CSV and a log
    """
    if model_path:
        load_model(model_path)

    # We removed the old mood layering. We'll just do a silent base or short.
    mood, polarity = analyze_text(text)
    logger.info(f"Overall mood: {mood} (polarity={polarity:.2f})")

    # Estimate durations for each sentence
    durations = estimate_sentence_durations(text, wpm=wpm)
    total_len = sum(durations)
    total_len = max(total_len, Config.BASE_LOOP_DURATION)

    # Start with silent base
    base = AudioSegment.silent(duration=total_len)

    # Now place normal SFX
    with_sfx = place_effects_with_timing(
        base, 
        text, 
        durations, 
        log_csv="train/soundscape_timeline.csv", 
        log_txt="train/soundscape_log.txt"
    )

    # If negativity or fear words, overlay a Markov fear stab pattern
    # This is optional. If you do want it for "fear" or negative:
    if "fear" in text.lower() or mood == "negative":
        stabs = generate_fear_stabs(length_ms=total_len, chord=220.0)
        with_sfx = with_sfx.overlay(stabs, position=0)

    with_sfx.export(output_path, format=Config.OUTPUT_FORMAT)
    logger.info(f"✅ Soundscape exported to {output_path}")

    # Finally, auto-generate a timeline visualization
    visualize_soundscape("train/soundscape_timeline.csv", "train/soundscape_timeline.png")
    logger.info("All done.")

def main():
    parser = argparse.ArgumentParser(description="Dynamic Soundscape Generator with Visualization")
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--output", default="soundscape_out.wav", help="Output WAV file")
    parser.add_argument("--wpm", type=int, default=150, help="Words per minute for read-time calculation")
    parser.add_argument("--model", default=None, help="Trained ML model path (sound_effect_model.pkl)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    create_soundscape_full(
        text,
        output_path=args.output,
        wpm=args.wpm,
        model_path=args.model
    )
    logger.info("Done.")

if __name__ == "__main__":
    main()
