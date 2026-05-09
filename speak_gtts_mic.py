"""
speak_gtts_mic.py - Text-to-Speech with virtual microphone output

This script converts text to speech and plays it through a virtual audio cable,
allowing you to simulate a human speaking in any app (Teams calls, recordings, etc.).

Requirements:
1. Install VB-Cable Virtual Audio Cable from https://vb-audio.com/Cable/
2. Set "CABLE Input" as your default playback device in Windows Sound settings
   OR set it as the recording device in your target app (Teams, etc.)
3. pip install gtts sounddevice soundfile numpy

Usage:
- Run the script, type text and press Enter to speak it
- Type 'q' and press Enter to quit
"""

from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import sys

# Try to find a virtual cable device
# Common virtual cable device names:
CABLE_DEVICE_NAMES = [
    "CABLE Output",       # VB-Cable
    "CABLE Input",        # VB-Cable
    "VoiceMeeter",        # Voicemeeter
    "Virtual Audio",      # Generic
    "Stereo Mix",         # Windows Stereo Mix (may work as loopback)
]

def find_virtual_cable_device():
    """Find a virtual audio cable device for output."""
    devices = sd.query_devices()
    
    # First try to find a known virtual cable device
    for device in devices:
        name = device['name']
        for cable_name in CABLE_DEVICE_NAMES:
            if cable_name.lower() in name.lower():
                if device['max_output_channels'] > 0:
                    print(f"  Found virtual cable: {name} (device {device['index']})")
                    return device['index']
    
    # If no virtual cable found, list available output devices
    print("\n  No virtual audio cable detected!")
    print("  Available output devices:")
    for device in devices:
        if device['max_output_channels'] > 0:
            print(f"    [{device['index']}] {device['name']}")
    print("\n  To use this as a virtual microphone, install VB-Cable from:")
    print("  https://vb-audio.com/Cable/")
    print("  Then set 'CABLE Input' as your playback device in Windows Sound settings.")
    print("  In your target app (Teams, etc.), select 'CABLE Output' as the microphone.\n")
    
    return None

def play_audio(file_path, device=None):
    """Play an audio file through the specified device(s).
    
    If a virtual cable device is specified, plays through both
    the virtual cable AND the default output device simultaneously,
    so you can hear the speech through your speakers.
    """
    data, samplerate = sf.read(file_path)
    
    if device is not None:
        # Play through both the virtual cable and the default output
        default_device = sd.default.device[1]  # default output device
        
        # Get info about both devices
        devices_info = sd.query_devices()
        default_name = devices_info[default_device]['name'] if default_device is not None else "default"
        cable_name = devices_info[device]['name']
        
        print(f"  Playing through: {cable_name} (virtual mic) + {default_name} (speakers)")
        
        # Play on both devices simultaneously using OutputStream
        # We need to create two output streams
        import threading
        
        def play_on_device(dev):
            sd.play(data, samplerate, device=dev)
            sd.wait()
        
        threads = []
        for dev in [device, default_device]:
            if dev is not None:
                t = threading.Thread(target=play_on_device, args=(dev,))
                t.start()
                threads.append(t)
        
        for t in threads:
            t.join()
    else:
        sd.play(data, samplerate)
        sd.wait()

def main():
    print("=== TTS Virtual Microphone ===")
    print("Type text and press Enter to speak it through the virtual mic.")
    print("Type 'q' and press Enter to quit.\n")
    
    # Find virtual cable device
    cable_device = find_virtual_cable_device()
    
    if cable_device is None:
        print("  Continuing with default audio output (speakers).")
        print("  Install VB-Cable for virtual microphone functionality.\n")
    
    while True:
        try:
            text = input("Input text: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        
        if text.strip().lower() == 'q':
            print("Exiting.")
            break
        
        if not text.strip():
            continue
        
        print(f"  Generating speech...")
        
        try:
            # Generate TTS audio
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                tmp_path = f.name
                tts.save(tmp_path)
            
            if cable_device is None:
                print(f"  Playing through speakers...")
            
            # Play the audio
            play_audio(tmp_path, device=cable_device)
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            print()
            
        except Exception as e:
            print(f"  Error: {e}\n")

if __name__ == "__main__":
    main()