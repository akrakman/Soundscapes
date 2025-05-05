## Soundscape Generator

This generator takes a story or narrative input (`.txt`) and produces a layered audio experience (`.wav`).

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Main libraries:
- `nltk`
- `spacy`
- `pydub`
- `matplotlib`
- `textblob`
- `numpy`
- `scikit-learn` (if using ML model)

Also download NLP assets:

```bash
python -m nltk.downloader stopwords wordnet omw-1.4
python -m spacy download en_core_web_sm
```

## Usage

### Basic

```bash
python soundscape.py --input story.txt --output soundscape.wav
```

### Optional Flags

```bash
  --wpm 150                        # Reading speed (words per minute)
  --model sound_effect_model.pkl   # Trained ML model path
```

> You **must train** your model using `train.py` beforehand to use the `--model` option.

---

## Directory Structure

```
.
├── soundscape.py
├── train.py
├── audio/
│   └── effect_*.wav              # All sound effects go here
├── train/
│   ├── soundscape_log.txt
│   ├── soundscape_timeline.csv
│   └── soundscape_timeline.png
|   └── story.txt                     # Your input story
```

## ML Model (Optional)

The system supports a trained ML model to predict sound labels from tokens:

- Use `train.py` to build and save a `sound_effect_model.pkl`.
- Use the `--model` flag during execution to enable it.

Without a model, the system will default to rule-based and synonym-matching logic.

## Output

After running, you’ll get:
- `soundscape.wav`: The final audio output.
- `train/soundscape_log.txt`: Sentence-by-sentence log of applied effects.
- `train/soundscape_timeline.csv`: Timestamped list of sound overlays.
- `train/soundscape_timeline.png`: Visual chart of the soundscape timeline.
