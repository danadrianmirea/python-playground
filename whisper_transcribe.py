
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")

audio_file = r"whisper_test.mp3"
output_file = "whisper_transcription.txt"

segments, info = model.transcribe(
    audio_file,
    beam_size=5,
    vad_filter=True
)

print(f"Detected language: {info.language}")

with open(output_file, "w", encoding="utf-8") as f:
    for segment in segments:
        minutes = int(segment.start // 60)
        seconds = segment.start % 60

        line = f"[{minutes:02d}:{seconds:05.2f}] {segment.text.strip()}"

        print(line)
        f.write(line + "\n")

print(f"\nSaved to {output_file}")
