#!/usr/bin/env python3
"""
USAGE (CLI):
  python main2.py --input story.txt --output soundscape.wav --model sound_effect_model.pkl

OR use GUI:
  python main2.py --gui
"""

import argparse
import logging
import os
import random
import sys
import pickle
import re
import time
from typing import List, Dict, Tuple

import nltk
import spacy
import numpy as np
from nltk.corpus import wordnet as wn, stopwords
from textblob import TextBlob
from pydub import AudioSegment, exceptions
from pydub.generators import Sine
import matplotlib.pyplot as plt

from gtts import gTTS
import tempfile

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QTextEdit, QMessageBox, QHBoxLayout
)
from PyQt5.QtCore import Qt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##################################################
#             LOAD SPAcy & CHECK NLTK
##################################################
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("SpaCy model 'en_core_web_sm' not found.\nInstall with:\n  python -m spacy download en_core_web_sm")
    sys.exit(1)

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
    AUDIO_DIR = "audio"            # Directory for audio files
    OUTPUT_FORMAT = "wav"
    BASE_LOOP_DURATION = 30_000    # 30 seconds minimum
    SENTIMENT_THRESHOLDS = {"positive": 0.15, "negative": -0.15}

    # Volume offsets
    VOLUME_EFFECTS = -12
    EFFECT_COOLDOWN_MS = 4000
    MAX_EFFECTS_PER_SENTENCE = 3

    # Fear crescendo
    FEAR_THRESHOLD = 2
    FEAR_MEMORY = 5
    FEAR_VOL_INCREMENT = 4
    FEAR_MAX_BOOST = 10

    # Simple mood-based background loops (no old dict)
    MOOD_BACKGROUND = {
        "positive": "background_positive_loop.wav",
        "negative": "background_creepy_loop.wav",
        "neutral":  "base_ambience_forest.wav"
    }

    # Ambient triggers
    AMBIENT_TRIGGERS = {
        "wind": "effect_wind.wav"
    }

    # TTS config
    TTS_ENABLED = True
    TTS_VOICE_GAIN = -20  # dB to apply to each token TTS chunk

    STOPWORDS = set(stopwords.words("english"))

##################################################
#         LABEL → AUDIO MAPPINGS
##################################################

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

# Fine-tuned synonyms for paper, letter, desk, etc.
SYNONYMS = {
    "ocean":         ["sea", "waves", "water"],
    "volcano":       ["eruption", "lava", "ash"],
    "forest":        ["woods", "trees"],
    "thunder":       ["lightning", "storm"],
    "rain":          ["shower", "drizzle", "downpour"],
    "wind":          ["breeze", "gust", "draft"],
    "dog_bark":      ["dog", "dogs", "cart", "dogcart"],
    # expanded for paper
    "paper_rustle":  ["paper", "papers", "book", "books", "desk", "desks", "document",
                      "documents", "casebook", "casebooks", "letter", "letters", "files", "file"]
}

AUDIO_CACHE = {}
MODEL_PIPELINE = None
TTS_CACHE = {}

##################################################
#  ML-BASED PREDICTION
##################################################

def load_model(model_path: str) -> None:
    global MODEL_PIPELINE
    if MODEL_PIPELINE is not None:
        return
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
    if MODEL_PIPELINE is None:
        return "generic_effect"

    probas = MODEL_PIPELINE.predict_proba([token_text])[0]
    label_idx = probas.argmax()
    confidence = probas[label_idx]
    predicted_label = MODEL_PIPELINE.classes_[label_idx]
    if confidence < 0.10:
        return "generic_effect"
    return predicted_label

##################################################
#  FALLBACK: Synonyms + WordNet
##################################################

def find_synonym_match(word: str) -> str:
    for label, syns in SYNONYMS.items():
        if word in syns:
            return label

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
    ml_label = predict_effect_label(token_text)
    if ml_label in LABEL_TO_EFFECTS:
        return LABEL_TO_EFFECTS[ml_label]

    fb = fallback_effect_label(token_text)
    if fb in LABEL_TO_EFFECTS:
        return LABEL_TO_EFFECTS[fb]

    return LABEL_TO_EFFECTS["generic_effect"]

##################################################
#  AUDIO UTILS
##################################################

def load_audio(filename: str) -> AudioSegment:
    if filename in AUDIO_CACHE:
        return AUDIO_CACHE[filename]

    path = os.path.join(Config.AUDIO_DIR, filename)
    if not os.path.exists(path):
        logger.warning(f"Missing audio: {filename}")
        seg = AudioSegment.silent(duration=2000)
        AUDIO_CACHE[filename] = seg
        return seg

    try:
        seg = AudioSegment.from_file(path)
        AUDIO_CACHE[filename] = seg
    except exceptions.CouldntDecodeError:
        logger.error(f"Failed to decode {filename}, returning silence.")
        seg = AudioSegment.silent(duration=2000)
        AUDIO_CACHE[filename] = seg

    return seg

def loop_audio_to_length(seg: AudioSegment, length_ms: int) -> AudioSegment:
    if len(seg) < 100:
        return AudioSegment.silent(duration=length_ms)
    repeats = (length_ms // len(seg)) + 1
    extended = seg * repeats
    return extended[:length_ms]

##################################################
# TTS PER-TOKEN (aligned)
##################################################

def generate_token_tts(word: str) -> AudioSegment:
    if not Config.TTS_ENABLED or not word.strip():
        return AudioSegment.silent(duration=5)

    word = word.lower()
    if word in TTS_CACHE:
        return TTS_CACHE[word]

    try:
        tts = gTTS(text=word, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
            temp_path = tf.name
        tts.save(temp_path)

        seg = AudioSegment.from_file(temp_path)
        seg = seg.apply_gain(Config.TTS_VOICE_GAIN)
        os.remove(temp_path)

        TTS_CACHE[word] = seg
        return seg

    except Exception as e:
        logger.warning(f"TTS generation failed for '{word}': {e}")
        return AudioSegment.silent(duration=5)



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
#  FEAR STABS
##################################################

def generate_fear_stabs(length_ms=30000, chord=220.0) -> AudioSegment:
    """
    A simple Markov-like chord stab approach for the 'fear' mood,
    or just for adding tension. We overlay random stabs if fear is high.
    """
    seg = AudioSegment.silent(duration=length_ms)
    ms_per_stab = 1000
    fade_ms = 100
    t = 0
    while t < length_ms:
        if random.random() < 0.20:
            wave = Sine(chord).to_audio_segment(duration=ms_per_stab)
            wave = wave.fade_in(fade_ms).fade_out(fade_ms).apply_gain(-18)
            seg = seg.overlay(wave, position=t)
        t += ms_per_stab
    return seg

##################################################
#  MOOD + AMBIENT
##################################################

def add_mood_loop(base: AudioSegment, mood: str, total_len: int) -> AudioSegment:
    if mood == "positive":
        loop_name = Config.MOOD_BACKGROUND["positive"]
    elif mood == "negative":
        loop_name = Config.MOOD_BACKGROUND["negative"]
    else:
        loop_name = Config.MOOD_BACKGROUND["neutral"]

    loop_seg = load_audio(loop_name)
    loop_seg = loop_audio_to_length(loop_seg, total_len)
    loop_seg = loop_seg.apply_gain(-10)
    return base.overlay(loop_seg)

def add_ambient_layers(base: AudioSegment, text: str, total_len: int) -> AudioSegment:
    lowered = base
    text_lower = text.lower()
    for k, loop_file in Config.AMBIENT_TRIGGERS.items():
        if k in text_lower:
            seg = load_audio(loop_file)
            seg = loop_audio_to_length(seg, total_len)
            seg = seg.apply_gain(-12)
            lowered = lowered.overlay(seg)
    return lowered

##################################################
# place_effects_with_timing: now includes token-level TTS
##################################################

def place_effects_with_timing(base_track: AudioSegment,
                              text: str,
                              durations: List[int],
                              log_csv: str = "train/soundscape_timeline.csv",
                              log_txt: str = "train/soundscape_log.txt") -> AudioSegment:
    """
    For each token:
      - Compute fraction-based offset
      - Generate TTS chunk for that token
      - Overlay TTS chunk
      - Overlay effect (if any)
    This ensures TTS & effect start at the same time.
    """
    import time
    final = base_track
    current_pos = 0
    sents = split_text_into_sentences(text)

    timeline_lines = ["timestamp_ms,effect,sentence_idx,token\n"]
    sentence_log = []

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
            raw = tok.text.strip().lower()
            word = re.sub(r"[^a-z0-9]+","", raw)

            # Fear memory
            if "fear" in word:
                fear_history.append(time.time())
            if len(fear_history) > Config.FEAR_MEMORY:
                fear_history = fear_history[-Config.FEAR_MEMORY:]
            if len(fear_history) >= Config.FEAR_THRESHOLD:
                heartbeat_boost = min(Config.FEAR_MAX_BOOST, heartbeat_boost + Config.FEAR_VOL_INCREMENT)
                fear_history.clear()

            fraction = j / max(1, (n_tokens-1))
            pos = current_pos + int(dur*fraction)

            # 1) TTS chunk for this token
            tts_seg = generate_token_tts(word)
            final = final.overlay(tts_seg, position=pos)

            # 2) Effects
            fx_files = get_effects_for_token(word)
            if not fx_files or fx_files == ["effect_generic.wav"]:
                continue

            if applied_this_sentence >= Config.MAX_EFFECTS_PER_SENTENCE:
                continue

            for fx_file in fx_files:
                if fx_file in last_applied:
                    if (pos - last_applied[fx_file])< Config.EFFECT_COOLDOWN_MS:
                        continue

                fx_audio = load_audio(fx_file)
                vol_gain = Config.VOLUME_EFFECTS
                if "heartbeat" in fx_file and heartbeat_boost>0:
                    vol_gain += heartbeat_boost
                vol_gain += random.randint(-2,2)
                fx_audio = fx_audio.fade_in(100).fade_out(100).apply_gain(vol_gain)
                final = final.overlay(fx_audio, position=pos)
                last_applied[fx_file] = pos
                applied_this_sentence += 1

                timeline_lines.append(f"{pos},{fx_file},{i+1},{word}\n")
                applied_effects.add(fx_file)

        sentence_log.append(f"Section {i+1}: \"{sent}\" → Applied {sorted(list(applied_effects))}")
        current_pos += dur

    with open(log_csv, "w", encoding="utf-8") as f:
        f.writelines(timeline_lines)
    logger.info(f"Timeline CSV: {log_csv}")

    with open(log_txt, "w", encoding="utf-8") as f:
        for line in sentence_log:
            f.write(line + "\n")
    logger.info(f"Sentence-level log: {log_txt}")

    return final

##################################################
#  VISUALIZE THE TIMELINE
##################################################

def visualize_soundscape(csv_path="train/soundscape_timeline.csv", out_fig="train/soundscape_timeline.png"):
    import csv
    if not os.path.exists(csv_path):
        logger.warning("No timeline CSV found; skipping visualization.")
        return
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        logger.warning("Timeline CSV empty; skipping.")
        return

    import matplotlib.pyplot as plt
    events = []
    track_map = {}
    track_id = 0

    for row in rows:
        ts = int(row["timestamp_ms"])
        fx = row["effect"]
        sidx = int(row["sentence_idx"])
        tok = row["token"]
        if fx not in track_map:
            track_map[fx] = track_id
            track_id += 1
        events.append((ts, fx, sidx, tok))

    events.sort(key=lambda x:x[0])
    data = []
    for (ts, fx, s, tok) in events:
        tid = track_map[fx]
        data.append((ts,tid,fx,tok))

    plt.figure(figsize=(12,6))
    for (ts, track, fx, tok) in data:
        bar_len = 800
        plt.plot([ts, ts+bar_len],[track, track], lw=6)

    yticks = []
    ylabels = []
    for f, t in track_map.items():
        yticks.append(t)
        ylabels.append(f)
    plt.yticks(yticks, ylabels)
    plt.xlabel("Time (ms)")
    plt.title("Soundscape Timeline")
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    logger.info(f"Soundscape timeline chart saved to {out_fig}")
    plt.close()

##################################################
#  CREATE FULL SOUNDSCAPE
##################################################

def create_soundscape_full(
    text: str,
    output_path: str = "soundscape_out.wav",
    wpm: int = 150,
    model_path: str = None
) -> None:
    """
    1) Load model if provided
    2) Analyze text for mood/polarity
    3) Create silent base
    4) Add mood-based loop
    5) Add ambient triggers
    6) Token-level TTS + effect placement
    7) If negative or 'fear', add Markov stabs
    """
    if model_path:
        load_model(model_path)

    mood, polarity = analyze_text(text)
    logger.info(f"Detected mood={mood}, polarity={polarity:.2f}")

    # compute total length from sentence durations
    durations = estimate_sentence_durations(text, wpm)
    total_len = sum(durations)
    total_len = max(total_len, Config.BASE_LOOP_DURATION)

    # 1) base silence
    base = AudioSegment.silent(duration=total_len)

    # 2) add mood
    base_mood = add_mood_loop(base, mood, total_len)

    # 3) add ambient
    base_amb = add_ambient_layers(base_mood, text, total_len)

    # 4) place token-level TTS + SFX
    sfx_mix = place_effects_with_timing(
        base_amb,
        text,
        durations,
        "train/soundscape_timeline.csv",
        "train/soundscape_log.txt"
    )

    # 5) fear stabs
    if mood=="negative" or "fear" in text.lower():
        stabs = generate_fear_stabs(length_ms=total_len, chord=220.0)
        sfx_mix = sfx_mix.overlay(stabs, position=0)

    # export
    sfx_mix.export(output_path, format=Config.OUTPUT_FORMAT)
    logger.info(f"✅ Soundscape exported to {output_path}")

    visualize_soundscape("train/soundscape_timeline.csv","train/soundscape_timeline.png")
    logger.info("All done.")

##################################################
#                GUI FRONT-END
##################################################
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLineEdit, QTextEdit, QMessageBox, QHBoxLayout, QLabel
)

class SoundscapeGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Soundscape (Token-level TTS)")
        self.resize(600,400)

        layout = QVBoxLayout()

        # input file
        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        open_file_btn = QPushButton("Open Text File...")
        open_file_btn.clicked.connect(self.on_open_file)
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(open_file_btn)
        layout.addLayout(file_layout)

        # model
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        open_model_btn = QPushButton("Open Model File...")
        open_model_btn.clicked.connect(self.on_open_model)
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(open_model_btn)
        layout.addLayout(model_layout)

        # direct text
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Or paste text here...")
        layout.addWidget(self.text_edit)

        # output
        out_layout = QHBoxLayout()
        self.out_edit = QLineEdit("soundscape_out.wav")
        out_btn = QPushButton("Output File...")
        out_btn.clicked.connect(self.on_open_output)
        out_layout.addWidget(self.out_edit)
        out_layout.addWidget(out_btn)
        layout.addLayout(out_layout)

        # WPM
        wpm_layout = QHBoxLayout()
        wpm_label = QLabel("WPM:")
        self.wpm_edit = QLineEdit("150")
        wpm_layout.addWidget(wpm_label)
        wpm_layout.addWidget(self.wpm_edit)
        layout.addLayout(wpm_layout)

        gen_btn = QPushButton("Generate Soundscape")
        gen_btn.clicked.connect(self.on_generate)
        layout.addWidget(gen_btn)

        self.setLayout(layout)

    def on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt)")
        if path:
            self.file_edit.setText(path)

    def on_open_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Model File", "", "Pickle Files (*.pkl)")
        if path:
            self.model_edit.setText(path)

    def on_open_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Output File", "", "WAV Files (*.wav)")
        if path:
            self.out_edit.setText(path)

    def on_generate(self):
        text_path = self.file_edit.text().strip()
        raw_text = self.text_edit.toPlainText().strip()
        model_path = self.model_edit.text().strip()
        out_path = self.out_edit.text().strip()
        if not out_path:
            out_path = "soundscape_out.wav"

        if text_path and os.path.exists(text_path):
            with open(text_path,"r",encoding="utf-8") as f:
                text = f.read()
        else:
            text = raw_text

        if not text.strip():
            QMessageBox.warning(self, "No Text", "Please provide text via file or direct input.")
            return

        wpm_val = 150
        try:
            wpm_val = int(self.wpm_edit.text())
        except:
            pass

        create_soundscape_full(
            text=text,
            output_path=out_path,
            wpm=wpm_val,
            model_path=model_path
        )
        QMessageBox.information(self, "Done", f"Soundscape saved to {out_path}")

def run_gui():
    app = QApplication(sys.argv)
    gui = SoundscapeGUI()
    gui.show()
    sys.exit(app.exec_())

##################################################
#                MAIN
##################################################

def main():
    parser = argparse.ArgumentParser(description="Dynamic Soundscape with Token-Level TTS")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--input", help="Path to input text file")
    parser.add_argument("--output", default="soundscape_out.wav", help="Output WAV file")
    parser.add_argument("--model", default=None, help="Trained ML model path (pkl)")
    parser.add_argument("--wpm", type=int, default=150, help="Words per minute")

    args = parser.parse_args()

    if args.gui:
        run_gui()
    else:
        if not args.input:
            print("Provide --input or use --gui")
            return
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)

        with open(args.input,"r",encoding="utf-8") as f:
            text_data = f.read()

        create_soundscape_full(
            text=text_data,
            output_path=args.output,
            wpm=args.wpm,
            model_path=args.model
        )
        logger.info("Done.")

if __name__=="__main__":
    main()
