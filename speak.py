import pyttsx3

engine = pyttsx3.init()
text = input("Enter text to speak: ")
engine.say(text)
engine.runAndWait()