import os
import whisper

# --- SETTINGS ---
folder = "clips"
model_name = "base"   # can use "small", "medium", or "large" for more accuracy

# --- LOAD MODEL ---
print(f"Loading Whisper model '{model_name}' ...")
model = whisper.load_model(model_name)

# --- PREPARE OUTPUT ---
transcript_path = os.path.join(folder, "transcript_auto.txt")

# --- GET FILES ---
wav_files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(".wav")]

# --- TRANSCRIBE EACH CLIP ---
with open(transcript_path, "w", encoding="utf-8") as out:
    for i, wav in enumerate(wav_files, 1):
        audio_path = os.path.join(folder, wav)
        print(f"[{i}/{len(wav_files)}] Transcribing {wav} ...")
        result = model.transcribe(audio_path, language="en")
        text = result["text"].strip()
        out.write(f"{wav}\t{text}\n")

print(f"\n✅ Automatic transcription done!")
print(f"Saved results to {transcript_path}")
