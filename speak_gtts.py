from gtts import gTTS
import subprocess

text = input("Input text: ")

# Use gTTS with English language
tts = gTTS(text=text, lang='en', slow=False)

# Save to out.mp3 and play it
tts.save("out.mp3")

# Play the audio file using the default system player
subprocess.Popen(['start', '', "out.mp3"], shell=True)