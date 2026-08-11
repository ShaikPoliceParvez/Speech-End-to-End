import sounddevice as sd

for sr in [16000,22050,24000,32000,44100,48000]:
    try:
        sd.check_output_settings(device=8, samplerate=sr)
        print(sr, "OK")
    except Exception as e:
        print(sr, e)





