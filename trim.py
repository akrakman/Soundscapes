from pydub import AudioSegment

audio = AudioSegment.from_file("audio/base_ambience_forest.mp3")  # Load the audio
trimmed_audio = audio[:5000]  # Trim to first 5000ms (5 seconds)
trimmed_audio.export("base_ambience_forest.mp3", format="mp3")  # Save the trimmed file
