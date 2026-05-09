from gtts import gTTS
import os
import tempfile

text = input("Input text: ")

# Use gTTS with Romanian language
tts = gTTS(text=text, lang='en', slow=False)

# Save to a temporary file and play it
with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
    tmp_path = f.name
    tts.save(tmp_path)

# Play the audio file
os.system(f'start "" "{tmp_path}"')