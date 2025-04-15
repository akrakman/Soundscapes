```markdown
# 🌀 Soundscape Generator

A production-ready **dynamic soundscape generator** that takes a story or narrative input (`.txt`) and produces a **layered audio experience** (`.wav`) — complete with:

- **Timeline visualization**
- **Token-triggered sound effects** (ML + rule-based)
- **Fear crescendo logic** (heartbeat intensifies with emotional tension)
- **Markov-style fear stabs** to evoke suspense

---

## 🔧 Features

1. **Visual Timeline Plot**  
   - Automatically generated `.png` showing sound effect overlays in time.
2. **Reliable Keyword Triggers**  
   - Hardcoded logic for consistent `dog_bark`, `paper_rustle`, and other critical cues.
3. **ML-Based Sound Effect Prediction**  
   - Trained model can dynamically classify tokens and predict suitable effects.
4. **Fear Crescendo Logic**  
   - Repeated mentions of fear intensify the `heartbeat` sound effect.
5. **Markov-Based Fear Stabs**  
   - Injects tension during negative emotional tone or fear-rich text.
6. **Fallback Using WordNet**  
   - Matches effects using synonyms for robustness.
7. **Production Logging**  
   - Soundscape timeline (`CSV`) and per-sentence effect logs (`TXT`).

---

## 📦 Requirements

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

---

## Usage

### Basic

```bash
python main2.py --input story.txt --output soundscape.wav
```

### Optional Flags

```bash
  --wpm 150                        # Reading speed (words per minute)
  --model sound_effect_model.pkl  # Trained ML model path
```

> You **must train** your model using `train.py` beforehand to use the `--model` option.

---

## 📁 Directory Structure

```
.
├── main2.py
├── train.py
├── audio/
│   └── effect_*.wav              # All sound effects go here
├── train/
│   ├── soundscape_log.txt
│   ├── soundscape_timeline.csv
│   └── soundscape_timeline.png
└── story.txt                     # Your input story
```

---

## 🎵 Supported Sound Effects

These must exist inside your `audio/` folder:

- `effect_dog_barking.wav`
- `effect_paper.wav`
- `effect_heartbeat.wav`
- `effect_rain.wav`, `effect_thunder.wav`, `effect_wind.wav`
- `effect_footsteps_wood.wav`, `effect_creaking_floor.wav`
- `effect_generic.wav` *(fallback)*  
…and others defined in `LABEL_TO_EFFECTS` in the code.

---

## 🤖 ML Model (Optional)

The system supports a trained ML model to predict sound labels from tokens:

- Use `train.py` to build and save a `sound_effect_model.pkl`.
- Use the `--model` flag during execution to enable it.

Without a model, the system will default to rule-based and synonym-matching logic.

---

## 📊 Output

After running, you’ll get:
- `soundscape.wav`: The final audio output.
- `train/soundscape_log.txt`: Sentence-by-sentence log of applied effects.
- `train/soundscape_timeline.csv`: Timestamped list of sound overlays.
- `train/soundscape_timeline.png`: Visual chart of the soundscape timeline.

---

## 🧠 Logic Highlights

- **Fear Triggering:** Repeated mentions of "fear" boost heartbeat sound intensity.
- **Cooldown System:** Prevents sound effect spam via time gating per effect.
- **Randomization:** Adds natural variation in gain and position.

---

## ✅ TODO (For Future Enhancements)

- [ ] Speaker-based voice synthesis
- [ ] Mood-based background loops
- [ ] Dynamic ambient layering
- [ ] GUI front-end

---

## 🧾 License

MIT License. Use freely with credit. Sound effects not included — ensure you have the right to use any `.wav` files you add.

---

## 👨‍💻 Author

Built for narrative sound design, horror storytelling, or game development.  
If you find it useful, drop a ⭐ or reach out!
```
