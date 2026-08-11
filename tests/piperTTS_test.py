from piper import PiperVoice
from pathlib import Path
import re

import numpy as np
import sounddevice as sd

voice = PiperVoice.load(
    Path(__file__).resolve().parent / "models" / "piper" / "te_IN-venkatesh-medium.onnx"
)

text = "నమస్కారం! నా పేరు షేక్ పర్వేజ్. ప్రస్తుతం నేను భారతీయ సాంకేతిక విద్యాసంస్థ హైదరాబాద్‌లో కంప్యూటర్ సైన్స్ అండ్ ఇంజినీరింగ్‌లో బి.టెక్ మూడవ సంవత్సరం చదువుతున్నాను.నాకు కృత్రిమ మేధస్సు, యంత్ర అభ్యాసం, సహజ భాషా ప్రాసెసింగ్ మరియు పూర్తి స్థాయి సాఫ్ట్‌వేర్ అభివృద్ధి వంటి రంగాలపై ఎంతో ఆసక్తి ఉంది. ప్రస్తుతం నేను ఒక బహుభాషా వాయిస్ అసిస్టెంట్‌ను అభివృద్ధి చేస్తున్నాను. ఈ ప్రాజెక్టులో స్పీచ్-టు-టెక్స్ట్, లార్జ్ లాంగ్వేజ్ మోడల్, టెక్స్ట్-టు-స్పీచ్ మరియు కంప్యూటర్ విజన్ వంటి ఆధునిక సాంకేతికతలను ఉపయోగిస్తున్నాను.నాకు ఇంగ్లీష్, హిందీ మరియు తెలుగు భాషల్లో మాట్లాడడం వస్తుంది. కొత్త సాంకేతికతలను నేర్చుకోవడం, ప్రయోగాలు చేయడం మరియు నిజ జీవిత సమస్యలకు ఉపయోగపడే పరిష్కారాలను రూపొందించడం నాకు చాలా ఇష్టం.భవిష్యత్తులో కృత్రిమ మేధస్సు మరియు సాఫ్ట్‌వేర్ ఇంజినీరింగ్ రంగాలలో పరిశోధనలు చేస్తూ, ప్రపంచవ్యాప్తంగా ఉపయోగపడే సాంకేతిక పరిష్కారాలను రూపొందించాలని నా లక్ష్యం.ధన్యవాదాలు!"


def sentence_stream(value):
    return (sentence.strip() for sentence in re.split(r"(?<=[.!?।])\s*", value) if sentence.strip())


def speak_stream(sentences):
    output_stream = None
    stream_config = None

    try:
        for sentence in sentences:
            print(f"Tarz: {sentence}")
            for chunk in voice.synthesize(sentence):
                config = (chunk.sample_rate, chunk.sample_channels)
                if config != stream_config:
                    if output_stream is not None:
                        output_stream.stop()
                        output_stream.close()
                    output_stream = sd.OutputStream(
                        samplerate=chunk.sample_rate,
                        channels=chunk.sample_channels,
                        dtype="float32",
                    )
                    output_stream.start()
                    stream_config = config

                audio = np.asarray(chunk.audio_float_array, dtype=np.float32)
                output_stream.write(audio.reshape(-1, chunk.sample_channels))
    finally:
        if output_stream is not None:
            output_stream.stop()
            output_stream.close()


speak_stream(sentence_stream(text))
print("Done!")