from playsound import playsound
import os

AUDIO_FILE = "test_tone.wav"
GST_PLUGIN_SCANNER_PATH = "/home/neville/miniconda3/libexec/gstreamer-1.0/gst-plugin-scanner"

def play_test_sound():
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file '{AUDIO_FILE}' not found.")
        print("Please run 'python create_test_tone.py' first to generate it.")
        return

    print(f"Attempting to play '{AUDIO_FILE}'...")
    try:
        # Set the GST_PLUGIN_SCANNER environment variable
        print(f"Setting GST_PLUGIN_SCANNER to: {GST_PLUGIN_SCANNER_PATH}")
        os.environ['GST_PLUGIN_SCANNER'] = GST_PLUGIN_SCANNER_PATH
        
        playsound(AUDIO_FILE)
        print(f"'{AUDIO_FILE}' playback finished (or started, if it runs in background).")
        print("Did you hear a sound?")
    except Exception as e:
        print(f"Error playing sound with playsound library: {e}")
        print("Please ensure all dependencies for playsound (like GStreamer, pygobject) are correctly installed and configured.")

if __name__ == "__main__":
    play_test_sound() 