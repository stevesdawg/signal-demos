import sys
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
wf = wave.open("record.wav", "w")
wf.setnchannels(CHANNELS)
wf.setsampwidth(pa.get_sample_size(WIDTH))
wf.setframerate(RATE)

def callback(in_data, frame_count, time_info, status):
    flag = pyaudio.paContinue
    wf.writeframes(in_data)
    print(status)
    print(time_info)
    if CLOSED:
        flag = pyaudio.paComplete
    return in_data, flag

stream = pa.open(
    format=WIDTH,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=False,
    stream_callback=callback
)

def close_all():
    stream.stop_stream()
    stream.close()
    pa.terminate()
    wf.close()

def main():
    global CLOSED
    DURATION = 5
    if len(sys.argv) == 2:
        try:
            DURATION = float(sys.argv[1])
        except:
            close_all()
            print("Usage: argument must be a number")
            return
    start_time = time.time()
    try:
        while stream.is_active() and not CLOSED:
            CLOSED = time.time() - start_time >= DURATION
            time.sleep(0.1)
    except KeyboardInterrupt:
        CLOSED = True
    close_all()

if __name__ == "__main__":
    main()
