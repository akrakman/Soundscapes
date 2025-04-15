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

    # fear stabs
    FEAR_STAB_PROB = 0.05
    FEAR_STAB_GAIN = -34

    # Volume offsets
    VOLUME_EFFECTS = -12
    EFFECT_COOLDOWN_MS = 4000
    MAX_EFFECTS_PER_SENTENCE = 3

    # Fear crescendo
    FEAR_THRESHOLD = 2
    FEAR_MEMORY = 5
    FEAR_VOL_INCREMENT = 4
    FEAR_MAX_BOOST = 10

    # Simple mood-based background loops
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
    TTS_VOICE_GAIN = -20  # dB to apply to each sentence TTS chunk

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

# Fine-tuned synonyms
SYNONYMS = {
    "ocean":         ["sea", "waves", "water"],
    "volcano":       ["eruption", "lava", "ash"],
    "forest":        ["woods", "trees"],
    "thunder":       ["lightning", "storm"],
    "rain":          ["shower", "drizzle", "downpour"],
    "wind":          ["breeze", "gust", "draft"],
    "dog_bark":      ["dog", "dogs", "dogcart"],
    "paper_rustle":  [
        "paper", "papers", "book", "books", "desk", "desks",
        "document", "documents", "casebook", "casebooks",
        "letter", "letters", "files", "file"
    ]
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
        logger.info(f"Loaded ML model from {model_path}")
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
    # Check our custom synonyms first
    for label, syns in SYNONYMS.items():
        if word in syns:
            return label

    # Fallback to WordNet lookups
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

def get_effects_for_token(token_text: str) -> Tuple[List[str], str]:
    """
    Return a tuple of:
    - list of audio files
    - method used: 'ml', 'synonym', 'wordnet', 'generic'
    """
    token_text = token_text.lower().strip()

    # Try ML model
    ml_label = predict_effect_label(token_text)
    if ml_label in LABEL_TO_EFFECTS:
        return LABEL_TO_EFFECTS[ml_label], "ml"

    # Try custom synonyms
    for label, syns in SYNONYMS.items():
        if token_text in syns and label in LABEL_TO_EFFECTS:
            return LABEL_TO_EFFECTS[label], "synonym"

    # Try WordNet
    for synset in wn.synsets(token_text):
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace("_", "").lower()
            if lemma_name in LABEL_TO_EFFECTS:
                return LABEL_TO_EFFECTS[lemma_name], "wordnet"

    # Default fallback
    return LABEL_TO_EFFECTS["generic_effect"], "generic"

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
#  TTS FOR SENTENCES
##################################################

def generate_sentence_tts(sentence: str) -> AudioSegment:
    """
    Generate TTS for a full sentence at once, which
    tends to sound more natural than per-token TTS.
    """
    if not Config.TTS_ENABLED or not sentence.strip():
        return AudioSegment.silent(duration=50)

    key = sentence.strip().lower()
    if key in TTS_CACHE:
        return TTS_CACHE[key]

    # Google TTS
    try:
        tts = gTTS(text=sentence, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
            temp_path = tf.name
        tts.save(temp_path)

        seg = AudioSegment.from_file(temp_path)
        seg = seg.apply_gain(Config.TTS_VOICE_GAIN)
        os.remove(temp_path)

        fade_ms = 50
        seg = seg.fade_in(fade_ms).fade_out(fade_ms)

        TTS_CACHE[key] = seg
        return seg

    except Exception as e:
        logger.warning(f"TTS generation failed for sentence '{sentence[:30]}...': {e}")
        return AudioSegment.silent(duration=100)

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
    # Filter out empty lines or weird blank segments
    return [s.text.strip() for s in doc.sents if s.text.strip()]

##################################################
#  FEAR STABS
##################################################

def generate_fear_stabs(length_ms=30000, chord=220.0) -> AudioSegment:
    """
    A simple approach for 'negative' or 'fear' mood:
    overlay random sine wave stabs.
    """
    seg = AudioSegment.silent(duration=length_ms)
    ms_per_stab = 800
    fade_ms = 100
    t = 0
    stab_times = []
    while t < length_ms:
        if random.random() < Config.FEAR_STAB_PROB:
            wave = Sine(chord).to_audio_segment(duration=ms_per_stab)
            wave = wave.fade_in(fade_ms).fade_out(fade_ms).apply_gain(Config.FEAR_STAB_GAIN)
            seg = seg.overlay(wave, position=t)
            stab_times.append(t)
        t += ms_per_stab
    return seg, stab_times

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
#  BUILD THE NARRATION TRACK
##################################################

def build_narration_track(
    text: str,
    timeline_csv_path="train/soundscape_timeline.csv",
    log_txt_path="train/soundscape_log.txt"
) -> AudioSegment:
    """
    - Split text into sentences.
    - For each sentence, generate TTS chunk.
    - Then scan token-by-token (spacy doc) to see if we should overlay SFX inside that chunk.
      * We'll place each SFX at a fraction of the TTS chunk length, matching the token index.
      * We fade them in/out and keep track of cooldown, etc.
    - Append each sentence's TTS+SFX chunk to a final 'narration_track'.
    - Write a timeline CSV as we go with real offsets in final track.
    """
    sents = split_text_into_sentences(text)
    narration_track = AudioSegment.silent(duration=0)

    timeline_lines = ["timestamp_ms,effect,source,sentence_idx,token\n"]
    sentence_log = []

    last_applied = {}
    fear_history = []
    heartbeat_boost = 0

    current_pos_in_narration = 0

    # Sentence loop
    for i, sent in enumerate(sents):
        # Generate TTS for this entire sentence
        tts_seg = generate_sentence_tts(sent)
        if len(tts_seg) < 50:
            # fallback
            tts_seg = AudioSegment.silent(duration=500)

        # We'll overlay SFX onto this "sentence_chunk" so it lines up with the voice
        sentence_chunk = tts_seg
        doc = nlp(sent)
        tokens = list(doc)
        n_tokens = len(tokens)

        # track which effects we've placed to avoid spamming
        applied_this_sentence = 0
        sentence_effects = []

        # keep a rolling count
        if "fear" in sent.lower():
            fear_history.append(time.time())
            if len(fear_history) > Config.FEAR_MEMORY:
                fear_history = fear_history[-Config.FEAR_MEMORY:]

        # For each token, place potential effect in the TTS chunk
        for j, tok in enumerate(tokens):
            raw = tok.text.strip()
            # Remove punctuation for effect matching
            word_clean = re.sub(r"[^a-z0-9]+","", raw.lower())

            if not word_clean:
                continue

            # Possibly increment heartbeat_boost if we see repeated "fear"
            if "fear" in word_clean:
                fear_history.append(time.time())
                if len(fear_history) >= Config.FEAR_THRESHOLD:
                    heartbeat_boost = min(Config.FEAR_MAX_BOOST, heartbeat_boost + Config.FEAR_VOL_INCREMENT)
                    fear_history.clear()

            fx_files, fx_source = get_effects_for_token(word_clean)
            if fx_files == ["effect_generic.wav"]:
                # means no strong match
                continue
            if applied_this_sentence >= Config.MAX_EFFECTS_PER_SENTENCE:
                continue

            # Place the effect in the chunk
            fraction = j / max(1, (n_tokens - 1))
            fx_offset = int(len(tts_seg) * fraction)

            for fx_file in fx_files:
                # check cooldown
                last_pos = last_applied.get(fx_file, -999999)
                if (current_pos_in_narration + fx_offset) - last_pos < Config.EFFECT_COOLDOWN_MS:
                    continue

                fx_audio = load_audio(fx_file)
                vol_gain = Config.VOLUME_EFFECTS
                if "heartbeat" in fx_file and heartbeat_boost > 0:
                    vol_gain += heartbeat_boost
                vol_gain += random.randint(-2,2)
                fx_audio = fx_audio.fade_in(50).fade_out(50).apply_gain(vol_gain)

                sentence_chunk = sentence_chunk.overlay(fx_audio, position=fx_offset)
                # record for timeline
                real_timestamp = current_pos_in_narration + fx_offset
                timeline_lines.append(f"{real_timestamp},{fx_file},{fx_source},{i+1},{word_clean}\n")
                sentence_effects.append(fx_file)

                last_applied[fx_file] = current_pos_in_narration + fx_offset
                applied_this_sentence += 1
                if applied_this_sentence >= Config.MAX_EFFECTS_PER_SENTENCE:
                    break

        # We have now built a TTS + SFX chunk for this sentence
        # Append to the final narration
        sentence_chunk = sentence_chunk.fade_in(20).fade_out(20)

        sentence_log.append(
            f"Section {i+1}: \"{sent}\" → Applied {sorted(list(set(sentence_effects)))}"
        )

        # Append
        start_of_chunk = len(narration_track)
        narration_track = narration_track + sentence_chunk
        end_of_chunk = len(narration_track)
        current_pos_in_narration = end_of_chunk

    with open(timeline_csv_path, "w", encoding="utf-8") as f:
        f.writelines(timeline_lines)
    logger.info(f"Timeline CSV saved: {timeline_csv_path}")

    with open(log_txt_path, "w", encoding="utf-8") as f:
        for line in sentence_log:
            f.write(line + "\n")
    logger.info(f"Sentence log saved: {log_txt_path}")

    return narration_track

##################################################
#  VISUALIZE THE TIMELINE
##################################################

def visualize_soundscape(
    csv_path="train/soundscape_timeline.csv",
    out_fig="train/soundscape_timeline.png",
    extra_events: Dict[str, List[int]] = None
):
    import csv

    if not os.path.exists(csv_path):
        logger.warning("No timeline CSV found; skipping visualization.")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows and not extra_events:
        logger.warning("No timeline events found; skipping.")
        return

    # Prepare effect events
    events = []
    track_map = {}
    track_id = 0

    for row in rows:
        ts = int(row["timestamp_ms"])
        fx = row["effect"]
        tok = row["token"]

        if fx not in track_map:
            track_map[fx] = track_id
            track_id += 1

        events.append((ts, fx, tok))

    # Sort and prepare plotting data
    events.sort(key=lambda x: x[0])
    data = [(ts, track_map[fx], fx, tok) for ts, fx, tok in events]

    # Setup plot
    plt.figure(figsize=(12, 6))

    for ts, track, fx, tok in data:
        bar_len = 800
        plt.plot([ts, ts + bar_len], [track, track], lw=6, label=fx if ts == data[0][0] else None)

    if extra_events:
        for label, timestamps in extra_events.items():
            track_map[label] = track_id
            for i, ts in enumerate(timestamps):
                bar_len = 800
                plt.plot(
                    [ts, ts + bar_len],
                    [track_id, track_id],
                    lw=4,
                    color="red",
                    label=label if i == 0 else ""
                )
            track_id += 1

    yticks = []
    ylabels = []
    for fx, t in sorted(track_map.items(), key=lambda x: x[1]):
        yticks.append(t)
        ylabels.append(fx)

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
    Overall process:
    1) Load ML model
    2) Analyze text for overall mood
    3) Build a 'narration track' that contains sentence-level TTS plus the SFX inside each sentence
    4) Determine total length of narration track
    5) Build a base track at least 30s or narration length
       with mood-based loop + ambient triggers
    6) Overlay the narration track
    7) overlay fear stabs if negative mood or 'fear' is in text
    8) Export + visualize
    """
    if model_path:
        load_model(model_path)

    mood, polarity = analyze_text(text)
    logger.info(f"Detected mood={mood}, polarity={polarity:.2f}")

    # Build the TTS + SFX track first
    narration_track = build_narration_track(text)

    # figure out total length
    total_len = max(len(narration_track), Config.BASE_LOOP_DURATION)

    # 1) base silence
    base = AudioSegment.silent(duration=total_len)

    # 2) add mood
    base_mood = add_mood_loop(base, mood, total_len)

    # 3) add ambient
    base_amb = add_ambient_layers(base_mood, text, total_len)

    # 4) overlay the narration track
    final_mix = base_amb.overlay(narration_track, position=0)

    # 5) fear stabs
    stab_times = []
    if mood=="negative" or "fear" in text.lower():
        stabs, stab_times = generate_fear_stabs(length_ms=total_len, chord=220.0)
        final_mix = final_mix.overlay(stabs, position=0)

    final_mix.export(output_path, format=Config.OUTPUT_FORMAT)
    logger.info(f"✅ Soundscape exported to {output_path}")

    visualize_soundscape(
        csv_path="train/soundscape_timeline.csv",
        out_fig="train/soundscape_timeline.png",
        extra_events={"fear_stab": stab_times}
    )
    logger.info("All done.")

##################################################
#                GUI FRONT-END
##################################################
class SoundscapeGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Soundscape (Improved Sentence-Level TTS)")
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
        wpm_label = QLabel("WPM (not heavily used now):")
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

        try:
            wpm_val = int(self.wpm_edit.text())
        except:
            wpm_val = 150

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
    parser = argparse.ArgumentParser(description="Dynamic Soundscape (Improved Sentence-Level TTS)")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--input", help="Path to input text file")
    parser.add_argument("--output", default="soundscape_out.wav", help="Output WAV file")
    parser.add_argument("--model", default=None, help="Trained ML model path (pkl)")
    parser.add_argument("--wpm", type=int, default=150, help="Words per minute (unused in new approach)")

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
