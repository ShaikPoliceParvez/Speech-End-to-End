from pydantic_settings import BaseSettings, SettingsConfigDict


class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    VOICE: str = "F2"       # SuperTonic speaker voice; F = female, M = male — try F1–F6 / M1–M2
    TTS_SPEED: float = 0.92  # playback speed: < 1 = slower/clearer, > 1 = faster

    # Sentence chunking — controls how tokens are grouped into speakable chunks
    TTS_MIN_CHARS: int = 40   # chunk must have at least this many chars before it speaks (avoids tiny blips)
    TTS_MIN_WORDS: int = 7    # OR at least this many words; either condition is enough to emit
    TTS_MAX_CHARS: int = 220  # hard upper limit; chunk is flushed immediately when exceeded
    TTS_MAX_WORDS: int = 36   # same hard limit by word count

    # First-chunk latency — how quickly TTS starts after the LLM begins responding
    TTS_FIRST_SENTENCE_IMMEDIATELY: bool = True   # speak the first sentence as soon as it ends
    TTS_FIRST_CHUNK_MIN_CHARS: int = 9999  # 9999 = disabled; don't emit mid-sentence before punctuation
    TTS_FIRST_CHUNK_MIN_WORDS: int = 9999  # 9999 = disabled; same — wait for a real sentence boundary
    TTS_FIRST_WORD_IMMEDIATELY: bool = True        # (unused in current flow) emit the very first word alone
    TTS_FIRST_SENTENCE_WORDWISE: bool = False      # word-by-word mode for the first sentence; off by default
    TTS_FIRST_SENTENCE_WORD_CHUNK_SIZE: int = 2    # words per chunk when wordwise mode is on

    # Pacing and punctuation splitting
    TTS_CHUNK_ON_MINOR_PUNCTUATION: bool = False  # also split at commas/semicolons; off keeps sentences whole
    TTS_LEAD_WORDS_IMMEDIATE: bool = True          # (unused unless lead_words path is enabled in app.py)
    TTS_LEAD_WORDS_COUNT: int = 2                  # how many lead words to emit when that mode is active
    # Fragments below BOTH thresholds are held and merged into the next sentence to avoid choppy single-word clips
    TTS_MIN_FORCE_CHARS: int = 15  # raise this to merge more short sentences together
    TTS_MIN_FORCE_WORDS: int = 3   # raise this to merge more short sentences together

    # Filler/preface — a short phrase played while the LLM is still thinking
    TTS_CONTEXT_PREFACE_ENABLED: bool = True   # play "Sure, let me check..." before the real response
    TTS_CONTEXT_PREFACE_RANDOM: bool = True    # pick filler randomly; False = always use the first one
    TTS_PREFACE_PACING: str = "slow"           # "slow" adds a slight pause; "normal" is standard pace
    TTS_PREFACE_MIN_WORDS: int = 6             # fillers shorter than this are skipped as too abrupt

    # Prefetch slots — how many sentences/chunks to prepare ahead of playback
    TTS_PREFETCH_TEXT: int = 2   # sentences queued for synthesis ahead of playback
    TTS_PREFETCH_AUDIO: int = 1  # audio chunks queued for playback ahead of speaker output


tts_settings = TTSSettings()

# ── Backend routing ──────────────────────────────────────────────────────────
# To add a language: add an entry here and download the Piper model.

TTS_LANGUAGE_BACKENDS: dict = {
    "en": {"backend": "supertonic"},
    "hi": {"backend": "supertonic"},
    "ne": {
        "backend": "piper",
        "model": "models/piper/ne_NP-chitwan-medium.onnx",
    },
    "te": {
        "backend": "piper",
        "model": "models/piper/te_IN-maya-medium.onnx",
    },
    "ml": {
        "backend": "piper",
        "model": "models/piper/ml_IN-arjun-medium.onnx",
    },
    "ar": {
        "backend": "piper",
        "model": "models/piper/ar_JO-kareem-medium.onnx",
    },
}

# ── Pronunciation overrides ──────────────────────────────────────────────────
# Telugu: Piper G2P splits Xున్నY clusters incorrectly; spacing fixes phoneme bounds.

TTS_PRONUNCIATION_MAP: dict = {
    "Tarz": "Taarz",
    "tarz": "taarz",
    "Parvez": "Par Vez",
    "parvez": "par vez",
    "Gemma": "Jemma",
    "gemma": "jemma",
    "Qwen": "Kwen",
    "qwen": "kwen",
    "Llama": "Lah-ma",
    "llama": "lah-ma",
    "Ollama": "Oh-lah-ma",
    "ollama": "oh-lah-ma",
    "Whisper": "Whisper",
    "whisper": "whisper",
    "ChatGPT": "Chat G P T",
    "chatgpt": "chat gee pee tee",
    "బాగున్నాను": "బాగు ఉన్నాను",
    "బాగున్నారు": "బాగు ఉన్నారు",
    "బాగున్నావు": "బాగు ఉన్నావు",
    "బాగుంది": "బాగు ఉంది",
    "బాగుందా": "బాగు ఉందా",
    "బాగ ఉన్నాను": "బాగు ఉన్నాను",
    "నేను బాగ ఉన్నాను": "నేను బాగు ఉన్నాను",
}

# ── Contextual filler phrases ────────────────────────────────────────────────

LANGUAGE_PREFACES: dict = {
    "en": {
        "greeting": ["Hello. It is really nice to hear from you.", "Hi there. Glad we are speaking right now."],
        "wellbeing_query": ["Thanks for asking.", "I am doing well, thank you."],
        "smalltalk": ["Glad to hear that.", "Nice!", "That sounds good."],
        "appreciation": ["Great!", "Awesome.", "Happy to hear that."],
        "generic": ["Sure, I can help with that.", "Alright, let me help you.", "Okay, let us go through it."],
        "answer": ["Sure, here is what I can tell you.", "Certainly, here is the answer.", "Alright, let me explain clearly."],
        "story": ["Sure, let us begin a story.", "Alright, here comes a story.", "Great, let us start the story."],
        "joke": ["Here is a joke for you.", "This one should make you smile."],
        "poem": ["Here's a poem.", "I hope you enjoy it."],
        "weather": ["Let me check.", "Here's the weather update."],
        "news": ["Here's what's happening.", "Let's take a look."],
        "camera": ["Opening the camera.", "Let me have a look."],
        "translation": ["Here's the translation."],
        "math": ["Let's calculate that."],
        "coding": ["Let's solve it.", "Here's the code."],
        "search": ["Looking into that."],
        "thanks": ["You're welcome!", "Happy to help!"],
        "goodbye": ["See you soon!", "Take care!"],
        "apology": ["No worries.", "That's alright."],
        "confirmation": ["Done.", "Consider it done."],
        "clarification": ["Could you clarify that?"],
        "fallback": ["Let me think about that."],
    },
    "hi": {
        "greeting": ["नमस्ते। आपसे फिर बात करके अच्छा लगा।", "नमस्कार। आपकी आवाज़ सुनकर अच्छा लगा।"],
        "wellbeing_query": ["पूछने के लिए धन्यवाद।", "मैं ठीक हूँ, धन्यवाद।"],
        "smalltalk": ["अच्छा लगा सुनकर।", "बहुत बढ़िया।", "यह अच्छा है।"],
        "appreciation": ["बहुत अच्छा!", "शानदार।", "यह सुनकर खुशी हुई।"],
        "generic": ["ज़रूर, मैं मदद करता हूँ।", "ठीक है, मैं मदद करता हूँ।"],
        "answer": ["ज़रूर, यह रहा जवाब।", "ठीक है, मैं साफ़-साफ़ बताता हूँ।"],
        "story": ["नमस्ते, कहानी शुरू करते हैं।"],
        "joke": ["यह रहा एक मज़ेदार चुटकुला।", "यह चुटकुला आपको पसंद आएगा।"],
        "poem": ["यह रही एक कविता।"],
        "weather": ["मौसम की जानकारी देखता हूँ।"],
        "news": ["यह रही ताज़ा जानकारी।"],
        "camera": ["कैमरा खोल रहा हूँ।"],
        "translation": ["यह रहा अनुवाद।"],
        "math": ["आइए गणना करते हैं।"],
        "coding": ["आइए इसे हल करते हैं।"],
        "search": ["देखता हूँ।"],
        "thanks": ["कोई बात नहीं!", "खुशी हुई मदद करके!"],
        "goodbye": ["फिर मिलेंगे!", "अपना ध्यान रखिए।"],
        "apology": ["कोई बात नहीं।"],
        "confirmation": ["हो गया।"],
        "clarification": ["क्या आप थोड़ा और स्पष्ट करेंगे?"],
        "fallback": ["एक क्षण सोचने दीजिए।"],
    },
    "te": {
        "greeting": ["నమస్కారం. మళ్లీ మాట్లాడటం చాలా ఆనందంగా ఉంది.", "నమస్కారం. మీతో మాట్లాడటం సంతోషంగా ఉంది."],
        "wellbeing_query": ["అడిగినందుకు ధన్యవాదాలు.", "నేను బాగు ఉన్నాను, ధన్యవాదాలు."],
        "smalltalk": ["వినడానికి సంతోషంగా ఉంది.", "చాలా సంతోషంగా ఉంది.", "అది మంచి విషయం."],
        "appreciation": ["అద్భుతం!", "సూపర్.", "అది విని సంతోషంగా ఉంది."],
        "generic": [
            "సరే, నేను పూర్తిగా సహాయం చేస్తాను.",
            "అలాగే, దీనిని కలిసి వివరంగా చూసేద్దాం.",
            "సరే, మీరు అడిగింది స్పష్టంగా చెబుతాను.",
        ],
        "answer": [
            "సరే, ఇది మీకు స్పష్టమైన సమాధానం.",
            "అలాగే, దీన్ని ఇప్పుడు సులభంగా వివరిస్తాను.",
            "చాలా బాగుంది, ఇప్పుడు ముఖ్యమైన విషయం చెబుతాను.",
        ],
        "story": ["సరే, కథను ప్రారంభిద్దాం.", "సరే, ఇప్పుడు ఒక కథ చెబుతాను."],
        "joke": ["మీ కోసం ఒక జోక్ ఉంది.", "ఇది వింటే నవ్వొస్తుంది."],
        "poem": ["ఇది ఒక కవిత.", "మీకు నచ్చుతుందని ఆశిస్తున్నాను."],
        "weather": ["సరే, వాతావరణ వివరాలు చెక్ చేస్తాను.", "ఇప్పుడు వాతావరణ సమాచారం చెబుతాను."],
        "news": ["సరే, తాజా వార్తలు చెబుతాను.", "ఇప్పుడు ఏముంది చూద్దాం."],
        "camera": ["సరే, కెమెరా తెరిచి చూస్తాను.", "అలాగే, ఒకసారి చూసి చెబుతాను."],
        "translation": ["సరే, ఇది అనువాదం.", "అలాగే, మీకు అనువదించి చెబుతాను."],
        "math": ["సరే, లెక్క చేద్దాం.", "అలాగే, దాన్ని దశలవారీగా గణిస్తాను."],
        "coding": ["సరే, దీన్ని పరిష్కరిద్దాం.", "అలాగే, కోడ్‌తో స్పష్టంగా చూపిస్తాను."],
        "search": ["సరే, దీనిని చూసి చెబుతాను.", "అలాగే, వివరాలు వెతికి చెబుతాను."],
        "thanks": ["ఎప్పుడైనా స్వాగతం, సహాయం చేయడం నాకు ఆనందం.", "మీకు ఉపయోగపడితే నిజంగా చాలా సంతోషంగా ఉంది."],
        "goodbye": ["సరే, మళ్లీ మాట్లాడే వరకు జాగ్రత్తగా ఉండండి.", "మళ్ళీ కలుద్దాం, మీ రోజు చాలా బాగుండాలి."],
        "apology": ["పర్లేదు."],
        "confirmation": ["సరే, మీరు చెప్పిన పని పూర్తిగా అయింది.", "అవును, పని విజయవంతంగా పూర్తయ్యింది."],
        "clarification": ["దయచేసి మరింత స్పష్టంగా ఒక్కసారి చెబుతారా?"],
        "fallback": ["ఒక్కసారి ఆలోచించి సరైన సమాధానం చెబుతాను."],
    },
    "ne": {
        "greeting": ["नमस्ते। फेरि तपाईंसँग कुरा गर्न पाउँदा खुशी लाग्यो।", "नमस्कार। अहिले तपाईंसँग जोडिन पाउँदा राम्रो लाग्यो।"],
        "wellbeing_query": ["सोध्नुभएकोमा धन्यवाद।", "म ठीक छु, धन्यवाद।"],
        "smalltalk": ["त्यो सुनेर खुशी लाग्यो।", "धेरै राम्रो कुरा हो।", "राम्रो छ, यसैगरी अघि बढौँ।"],
        "appreciation": ["एकदम राम्रो!", "धमाकेदार छ।", "यो सुन्दा साँच्चै खुशी लाग्यो।"],
        "generic": ["पक्का, म यसमा तपाईँलाई राम्रोसँग सहयोग गर्छु।", "ठिक छ, अब यसलाई सजिलैसँग बुझेर अघि बढौँ।"],
        "answer": ["ठिक छ, अब यसको स्पष्ट उत्तर दिन्छु।", "अवश्य, मुख्य कुरा अब सीधै बताउँछु।"],
        "story": ["ठिक छ, अब एउटा रमाइलो कथा सुरु गरौँ।", "अवश्य, यहाँ तपाईँका लागि एउटा कथा छ।"],
        "joke": ["ठिक छ, अब एउटा मजेदार जोक सुनाउँछु।", "यो जोक सुनेर पक्कै हाँसो आउँछ।"],
        "poem": ["यहाँ एउटा छोटो कविता प्रस्तुत गर्छु।", "यो कविता तपाईँलाई मन पर्ने आशा छ।"],
        "weather": ["ठिक छ, मौसमको जानकारी हेरेर बताउँछु।", "अहिलेको मौसम अपडेट तपाईँलाई दिन्छु।"],
        "news": ["ठिक छ, ताजा समाचारको सार बताउँछु।", "हेरौँ, अहिलेका मुख्य खबर के छन्।"],
        "camera": ["ठिक छ, क्यामेरा खोलेर एक पटक हेर्छु।", "अवश्य, हेरेपछि तपाईँलाई स्पष्ट बताउँछु।"],
        "translation": ["ठिक छ, अब यसको अनुवाद गरेर बताउँछु।", "अवश्य, यो सामग्री नेपालीमा मिलाएर दिन्छु।"],
        "math": ["ठिक छ, यो हिसाब अब चरणबद्ध गरौँ।", "अवश्य, अब यसलाई सजिलै गणना गरेर दिन्छु।"],
        "coding": ["ठिक छ, यो समस्या कोडबाट समाधान गरौँ।", "अवश्य, अब समाधानलाई चरणबद्ध रूपमा देखाउँछु।"],
        "search": ["ठिक छ, यसबारे आवश्यक विवरण खोजेर बताउँछु।", "अवश्य, म यो विषय तुरुन्तै जाँच गर्छु।"],
        "thanks": ["तपाईँलाई स्वागत छ, सहयोग गर्न पाउँदा खुशी लाग्यो।", "सधैं स्वागत छ, फेरि चाहियो भने भन्नुहोस्।"],
        "goodbye": ["ठिक छ, फेरि भेटौँला, राम्रोसँग बस्नुहोस्।", "आजका लागि यति, फेरि कुरा गरौँला।"],
        "apology": ["ठिक छ, कुनै समस्या छैन।"],
        "confirmation": ["हुन्छ, काम सफलतापूर्वक पूरा भयो।", "ठिक छ, यो कुरा अब सम्पन्न भयो।"],
        "clarification": ["कृपया यो भाग अलि स्पष्ट गरेर भन्नुहोस्।"],
        "fallback": ["एकछिन सोचेर राम्रो जवाफ दिन्छु।"],
    },
    "ml": {
        "greeting": ["നമസ്കാരം. വീണ്ടും നിങ്ങളോടു സംസാരിക്കുന്നത് സന്തോഷമാണ്.", "ഹലോ. നിങ്ങളുമായി സംസാരിക്കാൻ കഴിഞ്ഞത് സന്തോഷം."],
        "wellbeing_query": ["ചോദിച്ചതിന് നന്ദി.", "എനിക്ക് സുഖമാണ്, നന്ദി."],
        "smalltalk": ["അത് കേട്ട് സന്തോഷം.", "നല്ലതാണ്.", "അത് നല്ല കാര്യമാണ്."],
        "appreciation": ["വളരെ നല്ലത്!", "അദ്ഭുതം.", "അത് കേട്ട് സന്തോഷം."],
        "generic": ["ശരി, ഞാൻ സഹായിക്കാം.", "അങ്ങനെ ചെയ്യാം, നമുക്ക് നോക്കാം."],
        "answer": ["ശരി, ഇതാണ് ഉത്തരം.", "അങ്ങനെ ചെയ്യാം, ഞാൻ വ്യക്തമായി വിശദീകരിക്കാം."],
        "story": ["ശരി, കഥ തുടങ്ങാം.", "ഇപ്പോൾ ഒരു കഥ പറയാം."],
        "joke": ["നിങ്ങൾക്കായി ഒരു തമാശ പറയാം.", "ഇത് കേട്ടാൽ നിങ്ങൾ ചിരിക്കും."],
        "poem": ["ഇതാണ് ഒരു കവിത.", "നിങ്ങൾക്ക് ഇഷ്ടപ്പെടുമെന്ന് കരുതുന്നു."],
        "weather": ["ശരി, കാലാവസ്ഥ വിവരങ്ങൾ നോക്കാം.", "ഇപ്പോൾ കാലാവസ്ഥ അപ്ഡേറ്റ് പറയുന്നു."],
        "news": ["ശരി, പുതിയ വാർത്തകൾ പറയുന്നു.", "ഇപ്പോൾ എന്തുണ്ട് എന്ന് നോക്കാം."],
        "camera": ["ശരി, ക്യാമറ തുറന്ന് നോക്കാം.", "അങ്ങനെ ചെയ്യാം, ഒന്ന് കണ്ടു പറയുന്നു."],
        "translation": ["ശരി, ഇതാണ് വിവർത്തനം.", "അങ്ങനെ ചെയ്യാം, ഞാൻ ഇത് വിവർത്തനം ചെയ്ത് പറയുന്നു."],
        "math": ["ശരി, നമുക്ക് കണക്കാക്കാം.", "അങ്ങനെ ചെയ്യാം, ഘട്ടം ഘട്ടമായി കണക്കാക്കാം."],
        "coding": ["ശരി, ഇത് പരിഹരിക്കാം.", "അങ്ങനെ ചെയ്യാം, കോഡോടെ വിശദീകരിക്കാം."],
        "search": ["ശരി, ഞാൻ അത് തിരഞ്ഞ് പറയുന്നു.", "അങ്ങനെ ചെയ്യാം, വിവരങ്ങൾ കണ്ടെത്തി പറയുന്നു."],
        "thanks": ["സ്വാഗതം!", "സഹായിക്കാൻ സന്തോഷം!"],
        "goodbye": ["വീണ്ടും കാണാം!", "ശ്രദ്ധിക്കുക!"],
        "apology": ["പ്രശ്നമില്ല."],
        "confirmation": ["കഴിഞ്ഞു.", "ചെയ്തു."],
        "clarification": ["കുറച്ച് കൂടുതൽ വ്യക്തമാക്കാമോ?"],
        "fallback": ["ഒരു നിമിഷം ആലോചിക്കാം."],
    },
    "ar": {
        "greeting": ["مرحبًا. يسعدني جدًا التحدث معك من جديد.", "أهلًا. من الجميل سماعك والتواصل معك الآن."],
        "wellbeing_query": ["شكرًا لسؤالك.", "أنا بخير، شكرًا لك."],
        "smalltalk": ["سعيد بسماع ذلك.", "هذا جميل.", "رائع."],
        "appreciation": ["ممتاز!", "رائع جدًا.", "سعيد بذلك."],
        "generic": ["حسنًا، سأساعدك في ذلك.", "تمام، دعنا نراجع هذا معًا."],
        "answer": ["حسنًا، إليك الإجابة.", "تمام، سأوضح لك الأمر بشكل واضح."],
        "story": ["حسنًا، لنبدأ القصة.", "رائع، سأحكي لك قصة الآن."],
        "joke": ["إليك نكتة لطيفة.", "هذه نكتة قد تعجبك."],
        "poem": ["إليك قصيدة.", "أتمنى أن تنال إعجابك."],
        "weather": ["حسنًا، سأتفقد حالة الطقس الآن.", "تمام، إليك تحديث الطقس."],
        "news": ["حسنًا، إليك آخر الأخبار.", "تمام، دعنا نرى ما الجديد."],
        "camera": ["حسنًا، سأفتح الكاميرا الآن.", "تمام، دعني أنظر وأخبرك."],
        "translation": ["حسنًا، إليك الترجمة.", "تمام، سأترجم ذلك لك بوضوح."],
        "math": ["حسنًا، دعنا نحسب ذلك.", "تمام، سأحسبها خطوة بخطوة."],
        "coding": ["حسنًا، دعنا نحل ذلك.", "تمام، سأشرح لك الحل مع الكود."],
        "search": ["حسنًا، سأبحث عن ذلك الآن.", "تمام، سأجمع لك التفاصيل."],
        "thanks": ["على الرحب والسعة!", "سعيد بمساعدتك!"],
        "goodbye": ["إلى اللقاء!", "اعتنِ بنفسك!"],
        "apology": ["لا بأس."],
        "confirmation": ["تم.", "تم التنفيذ."],
        "clarification": ["هل يمكنك التوضيح أكثر؟"],
        "fallback": ["دعني أفكر في ذلك."],
    },
}
