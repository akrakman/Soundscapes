"""
An AI-driven soundscape generator that analyzes a given text excerpt 
(from The Adventure of the Speckled Band by Arthur Conan Doyle) and dynamically 
constructs an audio environment based on its mood, pacing, and context. 

It accomplishes this through NLP analysis, sentiment detection, 
machine learning-based sound effect prediction, and audio synthesis.
"""
import os
import logging
import random
import spacy
import numpy as np
from typing import Tuple, List, Dict
from textblob import TextBlob
from pydub import AudioSegment, exceptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class Config:
    AUDIO_DIR = "audio"
    OUTPUT_FORMAT = "wav"
    BASE_LOOP_DURATION = 30 * 1000  # 30 seconds
    VOLUME_ADJUSTMENTS = {
        "base": -5,  # dB adjustment for base track
        "effect": -10  # dB adjustment for effects
    }
    PACING_EFFECT_INTERVALS = {
        "fast": 5000,   # 5 seconds between effects
        "medium": 10000, # 10 seconds
        "slow": 15000   # 15 seconds
    }
    SENTIMENT_THRESHOLDS = {
        "positive": 0.15,
        "negative": -0.15
    }

MOOD_TO_AMBIENCE = {
    "positive": "base_ambience_forest.wav",
    "negative": "effect_dog_barking.wav",
    "neutral":  "effect_thunder.wav"
}

LABEL_TO_EFFECTS = {
    "storm": ["effect_thunder.wav", "effect_rain.wav"],
    "rain": ["effect_rain.wav"],
    "wind": ["effect_wind.wav"],
    "night_silence": ["effect_ominous_silence.wav"],
    "old_mansion": ["effect_creaking_floor.wav"],
    "footsteps_wood": ["effect_footsteps_wood.wav"],
    "footsteps_stone": ["effect_footsteps_stone.wav"],
    "running": ["effect_fast_footsteps.wav"],
    "train": ["effect_train.wav"],
    "paper_rustle": ["effect_paper.wav"],
    "heartbeat": ["effect_heartbeat.wav"],
    "dog_bark": ["effect_dog_barking.wav"],
    "city_ambience": ["effect_city_ambience.wav"],
    "generic_effect": ["effect_generic.wav"],
}



# Audio cache to avoid reloading files
AUDIO_CACHE = {}

################################################################
##############         UTILITY FUNCTIONS       #################
################################################################

def analyze_text(text: str) -> Tuple[str, str, List[str]]:
    """
    -overall mood using sentiment (positive/negative/neutral).
    -pacing (fast, medium, slow) based on sentence complexity.
    -extracts sound-related keywords.
    """
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > Config.SENTIMENT_THRESHOLDS["positive"]:
        mood = "positive"
    elif sentiment < Config.SENTIMENT_THRESHOLDS["negative"]:
        mood = "negative"
    else:
        mood = "neutral"

    doc = nlp(text)
    sentences = list(doc.sents)
    pacing = "medium"
    if sentences:
        avg_complexity = sum(
            len([token for token in sent if token.pos_ in ["VERB", "ADJ"]])
            for sent in sentences
        ) / len(sentences)
        if avg_complexity > 5:
            pacing = "fast"
        elif avg_complexity < 2:
            pacing = "slow"

    # holds keywords
    keywords = []
    for ent in doc.ents:
        if ent.label_ in ["LOC", "EVENT"]:
            keywords.append(ent.text.lower())
    return mood, pacing, list(set(keywords))


def validate_return_full_audio_path(filename: str) -> str:
    path = os.path.join(Config.AUDIO_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path

def load_audio(filename: str) -> AudioSegment:
    if filename not in AUDIO_CACHE:
        path = validate_return_full_audio_path(filename)
        try:
            AUDIO_CACHE[filename] = AudioSegment.from_file(path)
        except exceptions.CouldntDecodeError:
            logger.error(f"Failed to decode audio file: {path}")
            raise
    return AUDIO_CACHE[filename]

def create_base_ambience(mood: str) -> AudioSegment:
    """
    Generates a looping base ambience track based on the detected mood.
    selects an ambience audio file corresponding to the given mood,
    applies a volume adjustment, and loops if too short.
    """
    filename = MOOD_TO_AMBIENCE.get(mood, MOOD_TO_AMBIENCE["negative"])
    base = load_audio(filename)
    
    base = base.apply_gain(-20)
    
    # Loop to reach 30 seconds
    if len(base) < Config.BASE_LOOP_DURATION:
        loops = (Config.BASE_LOOP_DURATION // len(base)) + 1
        base = base * loops
    
    return base[:Config.BASE_LOOP_DURATION]  # Trim to exact duration


def apply_effects_and_pacing(base: AudioSegment, effects: List[str], pacing: str) -> AudioSegment:
    effect_interval = Config.PACING_EFFECT_INTERVALS[pacing]
    
    for effect_file in effects:
        try:
            fx = load_audio(effect_file).apply_gain(Config.VOLUME_ADJUSTMENTS["effect"])
        except Exception as e:
            logger.warning(f"Skipping effect {effect_file}: {str(e)}")
            continue
        
        # Calculate effect positions
        start = 0
        while start < len(base):
            position = start + random.randint(-1000, 1000)  # Add some randomness
            base = base.overlay(fx, position=max(0, position))
            start += effect_interval + random.randint(-2000, 2000)

    return base


################################################################
###############             AI                     #############
################################################################
def load_sound_effect_model(model_path: str = "sound_effect_model.pkl"):
    import pickle
    with open(model_path, "rb") as f:
        model_pipeline = pickle.load(f)
    return model_pipeline # pipeline object (vectorizer + classifier)

def predict_effect_label(text: str, model_pipeline) -> str:
    predicted_label = model_pipeline.predict([text])[0]
    return predicted_label # predicts a single sound effect label for the input text

def predict_with_fallback(text: str, model_pipeline, confidence_threshold=0.10):
    probs = model_pipeline.predict_proba([text])[0]

    if len(probs) == 0:
        return None

    max_prob = np.max(probs)
    predicted_label = model_pipeline.classes_[np.argmax(probs)] # Predicts a sound effect label

    return predicted_label if max_prob >= confidence_threshold else None

def match_effects_ml(text: str, model_pipeline) -> List[str]:
    """
    Predicts a sound effect label from text using the AI model.
    """
    predicted_label = predict_with_fallback(text, model_pipeline)

    if not predicted_label:
        logger.info(f"Confidence too low for text: '{text[:50]}...' => No effect applied.")
        return []

    if predicted_label in LABEL_TO_EFFECTS:
        return LABEL_TO_EFFECTS[predicted_label]

    logger.warning(f"Predicted label '{predicted_label}' not in LABEL_TO_EFFECTS. Skipping effect.")
    return []


def apply_section_effects_ml(base_track: AudioSegment, sections: List[str], model_pipeline, log_entries) -> AudioSegment:
    """
    Uses the ML model to predict sound effects for each SECTION and logs them.
    """
    SECTION_DURATION = 5000
    final_mix = base_track

    if model_pipeline is None:
        logger.warning("No model pipeline provided. Skipping ML-based section effects.")
        return base_track

    for i, sec in enumerate(sections):
        effect_files = match_effects_ml(sec, model_pipeline)

        if not effect_files:
            log_entries.append(f"Section {i+1} (Skipped): {sec[:50]}... (No confident prediction)")
            continue

        section_start = i * SECTION_DURATION
        log_entries.append(f"Section {i+1}: {sec[:50]}... → Applied {effect_files}")

        for e_file in effect_files:
            fx = load_audio(e_file).apply_gain(Config.VOLUME_ADJUSTMENTS["effect"])
            random_offset = random.randint(2000, 4000)
            position = section_start + random_offset
            final_mix = final_mix.overlay(fx, position=position)

    return final_mix


################################################################
##############       MAIN LOGIC                 ################
################################################################

def split_text_into_sections(text: str) -> List[str]:
    doc = nlp(text)
    sections = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sections # separate mood analysis for sentences

def analyze_mood_over_time(sections: List[str]) -> List[str]:
    mood_timeline = []
    for sec in sections:
        sentiment = TextBlob(sec).sentiment.polarity
        if sentiment > Config.SENTIMENT_THRESHOLDS["positive"]:
            mood_timeline.append("positive")
        elif sentiment < Config.SENTIMENT_THRESHOLDS["negative"]:
            mood_timeline.append("negative")
        else:
            mood_timeline.append("neutral")
    return mood_timeline # each section now has a sentiment like pos, neutral, neg

def create_dynamic_ambience(mood_timeline: List[str]) -> AudioSegment:
    if not mood_timeline:
        return AudioSegment.silent(duration=0)
    
    final_ambience = AudioSegment.silent(duration=0)
    
    for i, mood in enumerate(mood_timeline):
        # 1. Load ambience for this mood
        filename = MOOD_TO_AMBIENCE.get(mood, MOOD_TO_AMBIENCE["negative"])
        track = load_audio(filename)
        track = track.apply_gain(Config.VOLUME_ADJUSTMENTS["base"])
        
        # 2. trim/loop track to a certain section length
        # e.g. each section gets 10s or 15s
        desired_length = 10000
        if len(track) > desired_length:
            track = track[:desired_length]
        else:
            loops = (desired_length // len(track)) + 1
            track = track * loops
            track = track[:desired_length]
        
        # 3. Fade in/out to avoid abrupt transitions
        track = track.fade_in(2000).fade_out(2000)
        
        # 4. Append to final_ambience
        final_ambience += track
    
    return final_ambience # single AudioSegment


def estimate_read_time_ms(text: str, wpm=150) -> int:
    words = len(text.split())
    words_per_sec = wpm / 60.0
    reading_time_s = words / words_per_sec
    return int(reading_time_s * 1000)


def generate_soundscape(text: str, output_path: str = "holmes_soundscape.wav", model_pipeline=None) -> None:
    try:
        sections = split_text_into_sections(text)
        mood_timeline = analyze_mood_over_time(sections)
        dynamic_ambience = create_dynamic_ambience(mood_timeline)

        overall_mood, pacing, keywords = analyze_text(text)
        base_ambience = create_base_ambience(overall_mood)

        log_entries = []

        if model_pipeline is None:
            logger.warning("No AI model provided. Skipping ML-based effect selection.")
            effect_files_global = []
        else:
            effect_files_global = match_effects_ml(text, model_pipeline)

        if effect_files_global:
            base_with_effects = apply_effects_and_pacing(base_ambience, effect_files_global, pacing)
            while len(base_with_effects) < len(dynamic_ambience):
                base_with_effects = base_with_effects + base_with_effects  # Loop effect
            
            base_with_effects = base_with_effects[:len(dynamic_ambience)]
        else:
            logger.warning("No global effects predicted. No overall effect applied.")
            base_with_effects = base_ambience  # Keep only ambience if no global effect

        final_mix = apply_section_effects_ml(dynamic_ambience, sections, model_pipeline, log_entries)
        final_output = final_mix.overlay(base_with_effects)


        reading_ms = estimate_read_time_ms(text, wpm=150)
        final_duration = max(30_000, reading_ms)

        if len(final_output) < final_duration:
            loops_needed = (final_duration // len(final_output)) + 1
            final_output = final_output * loops_needed
        final_output = final_output[:final_duration]

        final_output.export(output_path, format=Config.OUTPUT_FORMAT)
        logger.info(f"Successfully created AI-driven Holmes soundscape at {output_path}")
        
        with open("soundscape_log.txt", "w") as log_file:
            log_file.write("\n".join(log_entries))
        logger.info("Logged applied sound effects in soundscape_log.txt")


    except Exception as e:
        logger.error(f"Soundscape generation failed: {str(e)}")
        raise

################################################################
###################### RUN SCRIPT ##############################
################################################################
if __name__ == "__main__":
    model_pipeline = load_sound_effect_model("sound_effect_model.pkl")

    # Test model predictions on lines from the excerpt
    test_sentences = [
        "It is fear, Mr. Holmes. It is terror.",
        "You have come in by train this morning, I see.",
        "You had a good drive in a dog-cart, along heavy roads."
    ]
    for sentence in test_sentences:
        label = predict_effect_label(sentence, model_pipeline)
        effects = match_effects_ml(sentence, model_pipeline)
        print(f"Text: {sentence}\nPredicted Label: {label}\nEffects: {effects}\n")

    # Read the entire excerpt from file
    with open("speckled_band_excerpt.txt", "r", encoding="utf-8") as file:
        text_excerpt = file.read()

    # Generate the final soundscape
    generate_soundscape(text_excerpt, "holmes_soundscape.wav", model_pipeline=model_pipeline)


    # “The Adventure of the Speckled Band” from 
    # The Adventures of Sherlock Holmes by Arthur Conan Doyle.
    # public commons
