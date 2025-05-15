import wave
import math
import struct

# Sound parameters
SAMPLE_RATE = 44100  # Hz
DURATION = 1         # seconds
FREQUENCY = 440      # Hz (A4 note)
AMPLITUDE = 32767    # Max amplitude for 16-bit audio
FILENAME = "test_tone.wav"

num_samples = int(DURATION * SAMPLE_RATE)
num_channels = 1  # Mono
sample_width = 2  # Bytes per sample (16-bit)
num_frames = num_samples

with wave.open(FILENAME, 'w') as wf:
    wf.setnchannels(num_channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(SAMPLE_RATE)
    wf.setnframes(num_frames)

    for i in range(num_frames):
        # Sine wave formula
        value = int(AMPLITUDE * math.sin(2 * math.pi * FREQUENCY * i / SAMPLE_RATE))
        # Pack as 16-bit signed integer
        data = struct.pack('<h', value)
        wf.writeframesraw(data)

print(f"'{FILENAME}' created successfully.") 