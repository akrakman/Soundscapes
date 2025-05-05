#!/usr/bin/env python3
"""
USAGE (CLI):
  python3 soundscape.py --input story.txt --output soundscape.wav --model sound_effect_model.pkl

OR use GUI:
  python3 soundscape.py --gui
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
from pydub import AudioSegment, exceptions, effects
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
#             LOAD spaCy & CHECK NLTK
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
    AUDIO_DIR = "audio"
    OUTPUT_FORMAT = "wav"
    BASE_LOOP_DURATION = 30_000
    SENTIMENT_THRESHOLDS = {"positive": 0.15, "negative": -0.15}
    FEAR_STAB_PROB = 0.05
    FEAR_STAB_GAIN = -34
    VOLUME_EFFECTS = -12
    EFFECT_COOLDOWN_MS = 4000
    MAX_EFFECTS_PER_SENTENCE = 2  # reduced for polish
    FEAR_THRESHOLD = 2
    FEAR_MEMORY = 5
    FEAR_VOL_INCREMENT = 4
    FEAR_MAX_BOOST = 10
    MOOD_BACKGROUND = {
        "positive": "background_positive_loop.wav",
        "negative": "background_creepy_loop.wav",
        "neutral":  "base_ambience_forest.wav"
    }
    AMBIENT_TRIGGERS = {"wind": "effect_wind.wav"}
    TTS_ENABLED = True
    TTS_VOICE_GAIN = -20
    STOPWORDS = set(stopwords.words("english"))

##################################################
#             ADDITIONAL UTILS
##################################################

def duck_under_voice(bed: AudioSegment,
                     voice: AudioSegment,
                     gain_during_voice: float = -8,
                     window_ms: int = 500,
                     fade_ms: int = 150) -> AudioSegment:
    """
    Lower the background whenever the voice is audible.
    """
    out = bed[:]
    pos = 0
    while pos < len(voice):
        chunk = voice[pos:pos+window_ms]
        if chunk.rms > 200:
            segment = out[pos:pos+window_ms].apply_gain(gain_during_voice)
            segment = segment.fade_in(fade_ms).fade_out(fade_ms)
            out = out.overlay(segment, position=pos)
        pos += window_ms
    return out

def high_pass(seg: AudioSegment, cutoff: int = 180) -> AudioSegment:
    """
    Roll off low frequencies under cutoff.
    """
    return seg.high_pass_filter(cutoff)

##################################################
#         LABEL → AUDIO MAPPINGS
##################################################

LABEL_TO_EFFECTS = {
    "storm": ["effect_thunder.wav", "effect_rain.wav"],
    "rain": ["effect_rain.wav"],
    "wind": ["effect_wind.wav"],
    "night_silence": ["effect_ominous_silence.wav"],
    "footsteps_wood": ["effect_footsteps_wood.wav"],
    "footsteps_stone": ["effect_footsteps_stone.wav"],
    "old_mansion": ["effect_creaking_floor.wav"],
    "running": ["effect_fast_footsteps.wav"],
    "train": ["effect_train.wav"],
    "paper_rustle": ["effect_paper.wav"],
    "heartbeat": ["effect_heartbeat.wav"],
    "dog_bark": ["effect_dog_barking.wav"],
    "generic_effect": ["effect_generic.wav"]
}

SYNONYMS = {
    "ocean": ["sea","waves","water"],
    "volcano": ["eruption","lava","ash"],
    "forest": ["woods","trees"],
    "thunder": ["lightning","storm"],
    "rain": ["shower","drizzle","downpour"],
    "wind": ["breeze","gust","draft"],
    "dog_bark": ["dog","dogs","dogcart"],
    "paper_rustle": [
        "paper","papers","book","books","desk","desks",
        "document","documents","casebook","casebooks",
        "letter","letters","files","file"
    ]
}

AUDIO_CACHE: Dict[str, AudioSegment] = {}
MODEL_PIPELINE = None
TTS_CACHE: Dict[str, AudioSegment] = {}

##################################################
#     ML-BASED PREDICTION
##################################################
def load_model(model_path: str) -> None:
    global MODEL_PIPELINE
    if MODEL_PIPELINE: return
    try:
        with open(model_path, "rb") as f:
            MODEL_PIPELINE = pickle.load(f)
        logger.info(f"Loaded ML model from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}. No ML predictions.")
        MODEL_PIPELINE = None

def predict_effect_label(token_text: str) -> str:
    if MODEL_PIPELINE is None:
        return "generic_effect"
    probas = MODEL_PIPELINE.predict_proba([token_text])[0]
    idx = probas.argmax(); conf = probas[idx]
    label = MODEL_PIPELINE.classes_[idx]
    return label if conf >= 0.10 else "generic_effect"

##################################################
#     FALLBACK: Synonyms + WordNet
##################################################
def find_synonym_match(word: str) -> str:
    for label, syns in SYNONYMS.items():
        if word in syns:
            return label
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            nm = lemma.name().replace("_","").lower()
            if nm in LABEL_TO_EFFECTS:
                return nm
    return None

def fallback_effect_label(token_text: str) -> str:
    m = find_synonym_match(token_text)
    return m if m and m in LABEL_TO_EFFECTS else "generic_effect"

##################################################
#      GET EFFECTS FOR TOKEN
##################################################
def get_effects_for_token(token_text: str) -> Tuple[List[str], str]:
    txt = token_text.lower().strip()
    ml_lbl = predict_effect_label(txt)
    if ml_lbl in LABEL_TO_EFFECTS:
        return LABEL_TO_EFFECTS[ml_lbl], "ml"
    syn = find_synonym_match(txt)
    if syn and syn in LABEL_TO_EFFECTS:
        return LABEL_TO_EFFECTS[syn], "synonym"
    for synset in wn.synsets(txt):
        for lemma in synset.lemmas():
            nm = lemma.name().replace("_","").lower()
            if nm in LABEL_TO_EFFECTS:
                return LABEL_TO_EFFECTS[nm], "wordnet"
    return LABEL_TO_EFFECTS["generic_effect"], "generic"

##################################################
#         AUDIO UTILS
##################################################
def load_audio(filename: str) -> AudioSegment:
    if filename in AUDIO_CACHE:
        return AUDIO_CACHE[filename]
    path = os.path.join(Config.AUDIO_DIR, filename)
    if not os.path.exists(path):
        logger.warning(f"Missing audio: {filename}")
        seg = AudioSegment.silent(duration=2000)
    else:
        try:
            seg = AudioSegment.from_file(path)
        except exceptions.CouldntDecodeError:
            logger.error(f"Failed to decode {filename}, using silence.")
            seg = AudioSegment.silent(duration=2000)
    AUDIO_CACHE[filename] = seg
    return seg

def loop_audio_to_length(seg: AudioSegment, length_ms: int) -> AudioSegment:
    if len(seg) < 100:
        return AudioSegment.silent(duration=length_ms)
    times = (length_ms // len(seg)) + 1
    ext = seg * times
    return ext[:length_ms]

##################################################
#      TTS FOR SENTENCES
##################################################
def generate_sentence_tts(sentence: str) -> AudioSegment:
    if not Config.TTS_ENABLED or not sentence.strip():
        return AudioSegment.silent(duration=50)
    key = sentence.strip().lower()
    if key in TTS_CACHE:
        return TTS_CACHE[key]
    try:
        tts = gTTS(text=sentence, lang='en', tld='com.au')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
            temp_path = tf.name
        tts.save(temp_path)
        seg = AudioSegment.from_file(temp_path).set_frame_rate(48000)
        seg = seg.apply_gain(Config.TTS_VOICE_GAIN)
        os.remove(temp_path)
        fade = 50
        seg = seg.fade_in(fade).fade_out(fade)
        TTS_CACHE[key] = seg
        return seg
    except Exception as e:
        logger.warning(f"TTS failed for '{sentence[:30]}...': {e}")
        return AudioSegment.silent(duration=100)

##################################################
#         TEXT ANALYSIS
##################################################
def analyze_text(text: str) -> Tuple[str, float]:
    pol = TextBlob(text).sentiment.polarity
    if pol > Config.SENTIMENT_THRESHOLDS["positive"]:
        return "positive", pol
    if pol < Config.SENTIMENT_THRESHOLDS["negative"]:
        return "negative", pol
    return "neutral", pol

def split_text_into_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]

##################################################
#             FEAR STABS
##################################################
def generate_fear_stabs(length_ms=30000, chord=220.0) -> Tuple[AudioSegment,List[int]]:
    seg = AudioSegment.silent(duration=length_ms)
    t = 0; times = []
    while t < length_ms:
        if random.random() < Config.FEAR_STAB_PROB:
            stab = Sine(chord).to_audio_segment(duration=800)
            stab = stab.fade_in(100).fade_out(100).apply_gain(Config.FEAR_STAB_GAIN)
            seg = seg.overlay(stab, position=t)
            times.append(t)
        t += 800
    return seg, times

##################################################
#      MOOD + AMBIENT LOOPS
##################################################
def add_mood_loop(base: AudioSegment, mood: str, total_len: int) -> AudioSegment:
    if mood=="positive":
        fname = Config.MOOD_BACKGROUND["positive"]
    elif mood=="negative":
        fname = Config.MOOD_BACKGROUND["negative"]
    else:
        fname = Config.MOOD_BACKGROUND["neutral"]
    loop = load_audio(fname)
    loop = high_pass(loop, cutoff=180)
    loop = loop_audio_to_length(loop, total_len).apply_gain(-10)
    return base.overlay(loop)

def add_ambient_layers(base: AudioSegment, text: str, total_len: int) -> AudioSegment:
    out = base
    tl = text.lower()
    for key, fname in Config.AMBIENT_TRIGGERS.items():
        if key in tl:
            seg = load_audio(fname)
            seg = high_pass(seg, cutoff=180)
            seg = loop_audio_to_length(seg, total_len).apply_gain(-12)
            out = out.overlay(seg)
    return out

##################################################
#      BUILD NARRATION TRACK
##################################################
def build_narration_track(
    text: str,
    timeline_csv_path="train/soundscape_timeline.csv",
    log_txt_path="train/soundscape_log.txt"
) -> AudioSegment:
    sents = split_text_into_sentences(text)
    narration = AudioSegment.silent(0)
    timeline = ["timestamp_ms,effect,source,sentence_idx,token\n"]
    log = []
    last_applied = {}
    fear_hist = []
    heartbeat_boost = 0
    pos_in_narr = 0

    for idx, sent in enumerate(sents):
        tts_seg = generate_sentence_tts(sent)
        if len(tts_seg) < 50:
            tts_seg = AudioSegment.silent(500)
        chunk = tts_seg
        doc = nlp(sent); toks = list(doc)
        applied = 0; effects_this = []

        if "fear" in sent.lower():
            fear_hist.append(time.time())
            if len(fear_hist)>Config.FEAR_MEMORY:
                fear_hist=fear_hist[-Config.FEAR_MEMORY:]

        for j, tok in enumerate(toks):
            raw = tok.text.strip()
            clean = re.sub(r"[^a-z0-9]+","", raw.lower())
            if not clean: continue
            if "fear" in clean:
                fear_hist.append(time.time())
                if len(fear_hist)>=Config.FEAR_THRESHOLD:
                    heartbeat_boost = min(Config.FEAR_MAX_BOOST, heartbeat_boost + Config.FEAR_VOL_INCREMENT)
                    fear_hist.clear()
            fx_files, src = get_effects_for_token(clean)
            if fx_files==["effect_generic.wav"]: continue
            if applied>=Config.MAX_EFFECTS_PER_SENTENCE: break
            frac = j/max(1,(len(toks)-1)); offset = int(len(tts_seg)*frac)
            for fx in fx_files:
                last = last_applied.get(fx,-1e9)
                if (pos_in_narr+offset)-last < Config.EFFECT_COOLDOWN_MS:
                    continue
                fx_audio = load_audio(fx)
                fx_audio = fx_audio.pan(random.uniform(-0.6,0.6))
                vol = Config.VOLUME_EFFECTS + random.randint(-2,2)
                if "heartbeat" in fx and heartbeat_boost>0:
                    vol += heartbeat_boost
                fx_audio = fx_audio.apply_gain(vol).fade_in(50).fade_out(50)
                chunk = chunk.overlay(fx_audio, position=offset)
                timeline.append(f"{pos_in_narr+offset},{fx},{src},{idx+1},{clean}\n")
                effects_this.append(fx)
                last_applied[fx] = pos_in_narr+offset
                applied+=1
                if applied>=Config.MAX_EFFECTS_PER_SENTENCE:
                    break
        chunk = chunk.fade_in(20).fade_out(20)
        log.append(f"Section {idx+1}: \"{sent}\" → {sorted(set(effects_this))}")
        start = len(narration)
        narration += chunk
        pos_in_narr = len(narration)

    os.makedirs(os.path.dirname(timeline_csv_path), exist_ok=True)
    with open(timeline_csv_path,"w",encoding="utf-8") as f:
        f.writelines(timeline)
    logger.info(f"Timeline CSV saved: {timeline_csv_path}")
    with open(log_txt_path,"w",encoding="utf-8") as f:
        f.write("\n".join(log))
    logger.info(f"Sentence log saved: {log_txt_path}")
    return narration

##################################################
#     VISUALIZE THE TIMELINE
##################################################
def visualize_soundscape(
    csv_path="train/soundscape_timeline.csv",
    out_fig="train/soundscape_timeline.png",
    extra_events: Dict[str, List[int]] = None
):
    import csv
    if not os.path.exists(csv_path):
        logger.warning("No timeline CSV; skipping visualization.")
        return
    rows = []
    with open(csv_path,"r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows and not extra_events:
        logger.warning("No events; skipping.")
        return
    events=[]
    track_map={}; tid=0
    for r in rows:
        ts=int(r["timestamp_ms"]); fx=r["effect"]
        if fx not in track_map:
            track_map[fx]=tid; tid+=1
        events.append((ts,fx,r["token"]))
    events.sort(key=lambda x:x[0])
    data=[(ts,track_map[fx],fx,tok) for ts,fx,tok in events]
    plt.figure(figsize=(12,6))
    for ts,track,fx,tok in data:
        plt.plot([ts,ts+800],[track,track],lw=6,label=fx if ts==data[0][0] else None)
    if extra_events:
        for lbl,tss in extra_events.items():
            track_map[lbl]=tid
            for i,ts in enumerate(tss):
                plt.plot([ts,ts+800],[tid,tid],lw=4,color="red",
                         label=lbl if i==0 else "")
            tid+=1
    yticks=[]; ylabels=[]
    for fx,t in sorted(track_map.items(),key=lambda x:x[1]):
        yticks.append(t); ylabels.append(fx)
    plt.yticks(yticks,ylabels)
    plt.xlabel("Time (ms)"); plt.title("Soundscape Timeline")
    plt.grid(True,axis="x",linestyle="--",alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.savefig(out_fig,dpi=150)
    plt.close()
    logger.info(f"Timeline chart saved to {out_fig}")

##################################################
#  CREATE FULL SOUNDSCAPE
##################################################
def create_soundscape_full(
    text: str,
    output_path: str = "soundscape_out.wav",
    wpm: int = 150,
    model_path: str = None
) -> None:
    if model_path:
        load_model(model_path)
    mood, polarity = analyze_text(text)
    logger.info(f"Detected mood={mood}, polarity={polarity:.2f}")

    narration = build_narration_track(text)
    total_len = max(len(narration), Config.BASE_LOOP_DURATION)

    base = AudioSegment.silent(duration=total_len)
    base_mood = add_mood_loop(base, mood, total_len)
    base_amb = add_ambient_layers(base_mood, text, total_len)

    base_amb = duck_under_voice(base_amb, narration)

    # overlay narration
    final_mix = base_amb.overlay(narration, position=0)

    # fear stabs for negative / fear text
    stab_times = []
    if mood=="negative" or "fear" in text.lower():
        stabs, stab_times = generate_fear_stabs(length_ms=total_len)
        final_mix = final_mix.overlay(stabs, position=0)

    # master normalize + gentle compressor
    final_mix = effects.normalize(final_mix, headroom=1.0)
    final_mix = effects.compress_dynamic_range(
        final_mix,
        threshold=-20.0, ratio=4.0,
        attack=5, release=60
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final_mix.export(
        output_path,
        format=Config.OUTPUT_FORMAT,
        codec="pcm_s32le",
        parameters=["-ar", "48000", "-ac", "2"]
    )
    logger.info(f"Soundscape exported to {output_path}")

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
        self.setWindowTitle("Dynamic Soundscape (Enhanced)")
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
        wpm_label = QLabel("WPM (unused):")
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
        if path: self.file_edit.setText(path)

    def on_open_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Model File", "", "Pickle Files (*.pkl)")
        if path: self.model_edit.setText(path)

    def on_open_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Output File", "", "WAV Files (*.wav)")
        if path: self.out_edit.setText(path)

    def on_generate(self):
        text_path = self.file_edit.text().strip()
        raw_text = self.text_edit.toPlainText().strip()
        model_path = self.model_edit.text().strip()
        out_path = self.out_edit.text().strip() or "soundscape_out.wav"
        if text_path and os.path.exists(text_path):
            with open(text_path,"r",encoding="utf-8") as f: text = f.read()
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

def main():
    parser = argparse.ArgumentParser(description="Dynamic Soundscape (Enhanced)")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--input", help="Path to input text file")
    parser.add_argument("--output", default="soundscape_out.wav", help="Output WAV file")
    parser.add_argument("--model", default=None, help="Trained ML model path (pkl)")
    parser.add_argument("--wpm", type=int, default=150, help="Words per minute (unused)")
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
