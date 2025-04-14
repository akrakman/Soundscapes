from pydub import AudioSegment

audio = AudioSegment.from_file("audio/base_ambience_forest.wav")
trimmed_audio = audio[:5000]  # Trim to 5 seconds
trimmed_audio.export("base_ambience_forest.wav", format="wav")
