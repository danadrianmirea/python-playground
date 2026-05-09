"""
speak_gtts_mic.py - Text-to-Speech with virtual microphone output

This script converts text to speech and plays it through a virtual audio cable,
allowing you to simulate a human speaking in any app (Teams calls, recordings, etc.).

Now uses edge-tts (Microsoft Edge TTS) which supports both male and female voices.

Requirements:
1. Install VB-Cable Virtual Audio Cable from https://vb-audio.com/Cable/
2. Set "CABLE Input" as your default playback device in Windows Sound settings
   OR set it as the recording device in your target app (Teams, etc.)
3. pip install edge-tts sounddevice soundfile numpy

Usage:
- Run the script, type text and press Enter to speak it
- Type 'v' and press Enter to change voice (male/female)
- Type 'q' and press Enter to quit
"""

import edge_tts
import asyncio
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

# Will be populated dynamically at startup from edge-tts --list-voices
VOICES = {}
current_voice_key = None


async def load_voices():
    """Fetch available voices from edge-tts and populate the VOICES dict.
    
    Includes English (en-*) and Romanian (ro-*) voices for a manageable menu.
    Returns the key of the first available voice, or None if no voices found.
    """
    global VOICES, current_voice_key
    
    print("  Loading available voices from edge-tts...")
    all_voices = await edge_tts.list_voices()
    
    # Filter to English and Romanian voices
    filtered_voices = [v for v in all_voices if v['Locale'].startswith('en-') or v['Locale'].startswith('ro-')]
    
    if not filtered_voices:
        print("  No English or Romanian voices found! Falling back to all available voices.")
        filtered_voices = all_voices
    
    VOICES.clear()
    for i, voice in enumerate(filtered_voices, 1):
        key = str(i)
        short_name = voice['ShortName']
        gender = voice['Gender']
        # Build a concise display name: "Gender (ShortName)" e.g. "Female (en-US-JennyNeural)"
        display_name = f"{gender} ({short_name})"
        VOICES[key] = {
            "name": display_name,
            "voice": short_name,
        }
    
    # Set default to first voice
    if VOICES:
        current_voice_key = "1"
        print(f"  Loaded {len(VOICES)} voices (English + Romanian).")
    else:
        print("  WARNING: No voices available from edge-tts!")
    
    return current_voice_key


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
        
        # Play on both devices simultaneously using threads
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


async def generate_speech(text, voice, retries=2):
    """Generate TTS audio using edge-tts and save to a temp file.
    
    Retries up to `retries` times if edge-tts fails to receive audio
    (which can happen when switching voices).
    """
    for attempt in range(1, retries + 1):
        try:
            tts = edge_tts.Communicate(text=text, voice=voice)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                tmp_path = f.name
            
            await tts.save(tmp_path)
            return tmp_path
        except Exception as e:
            # Clean up temp file if it was created
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            if attempt < retries:
                print(f"  Retrying ({attempt}/{retries})...")
                await asyncio.sleep(0.5)
            else:
                raise e


async def test_voice(voice):
    """Test if a voice works by generating a short phrase."""
    try:
        await generate_speech("test", voice, retries=1)
        return True
    except Exception:
        return False


def show_voice_menu():
    """Display available voices and let the user pick one."""
    global current_voice_key
    
    if not VOICES:
        print("\n  No voices available. Try restarting the script.")
        return
    
    print("\n  Available voices:")
    for key, info in VOICES.items():
        marker = " <-- current" if key == current_voice_key else ""
        print(f"    [{key}] {info['name']}{marker}")
    
    choice = input(f"  Select voice (1-{len(VOICES)}): ").strip()
    if choice in VOICES:
        # Test the selected voice before switching
        voice = VOICES[choice]['voice']
        print(f"  Testing voice {VOICES[choice]['name']}...")
        works = asyncio.run(test_voice(voice))
        if works:
            current_voice_key = choice
            print(f"  Voice set to: {VOICES[choice]['name']}")
        else:
            print(f"  Voice '{VOICES[choice]['name']}' is not responding. Keeping current voice.")
    else:
        print(f"  Invalid choice, keeping current voice: {VOICES[current_voice_key]['name']}")
    print()


def main():
    global current_voice_key
    
    print("=== TTS Virtual Microphone (edge-tts) ===")
    print("Type text and press Enter to speak it through the virtual mic.")
    print("Type 'v' and press Enter to change voice.")
    print("Type 'q' and press Enter to quit.\n")
    
    # Load voices dynamically from edge-tts
    asyncio.run(load_voices())
    
    if not VOICES:
        print("  ERROR: No voices available. Exiting.")
        return
    
    # Find virtual cable device
    cable_device = find_virtual_cable_device()
    
    if cable_device is None:
        print("  Continuing with default audio output (speakers).")
        print("  Install VB-Cable for virtual microphone functionality.\n")
    
    # Test the default voice on startup
    default_voice = VOICES[current_voice_key]['voice']
    print(f"  Testing default voice ({VOICES[current_voice_key]['name']})...")
    works = asyncio.run(test_voice(default_voice))
    if not works:
        # Fall back to first working voice
        print(f"  Default voice not responding, searching for working voice...")
        for key, info in VOICES.items():
            test_result = asyncio.run(test_voice(info['voice']))
            if test_result:
                current_voice_key = key
                print(f"  Using voice: {info['name']}")
                break
    else:
        print(f"  Voice OK: {VOICES[current_voice_key]['name']}")
    print()
    
    while True:
        try:
            text = input("Input text: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        
        if text.strip().lower() == 'q':
            print("Exiting.")
            break
        
        if text.strip().lower() == 'v':
            show_voice_menu()
            continue
        
        if not text.strip():
            continue
        
        voice = VOICES[current_voice_key]['voice']
        voice_name = VOICES[current_voice_key]['name']
        print(f"  Generating speech ({voice_name})...")
        
        try:
            # Generate TTS audio using edge-tts (async)
            tmp_path = asyncio.run(generate_speech(text, voice))
            
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