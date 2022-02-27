import time
import wave

import pyaudio

import config

WIDTH = config.WIDTH
CHANNELS = config.CHANNELS
RATE = config.RATE
CHUNK = config.CHUNK

CLOSED = False

pa = pyaudio.PyAudio()
wf = wave.open("record.wav", "rb")

def callback(in_data, frame_count, time_info, status):
    flag = pyaudio.paContinue
    data = wf.readframes(frame_count)
    if CLOSED:
        flag = pyaudio.paComplete
    return data, flag

stream = pa.open(
    format=WIDTH,
    channels=CHANNELS,
    rate=RATE,
    input=False,
    output=True,
    stream_callback=callback
)

def close_all():
    stream.stop_stream()
    stream.close()
    pa.terminate()
    wf.close()

def main():
    global CLOSED
    try:
        while stream.is_active() and not CLOSED:
            time.sleep(0.1)
    except KeyboardInterrupt:
        CLOSED = True
    close_all()

if __name__ == "__main__":
    main()
