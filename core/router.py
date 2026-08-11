import re


class Router:

    def __init__(self):
        # Semantic phrase banks (multilingual), scored by intent groups.
        self.vision_action_phrases = [
            "what do you see", "look at this", "see this", "describe this image",
            "describe this picture", "look around", "capture image",
            "क्या दिख रहा है", "क्या देख रहे हो", "देखो", "जरा देखो", "ज़रा देखो",
            "ఏం కనిపిస్తోంది", "ఇది చూడు", "చూసి చెప్పు", "కెమెరా తెరువు",
            "tum kya dekh rahe ho", "dekh ke bata", "jara dekh", "zara dekh",
            "em kanipistundi", "idi choodu", "choosi cheppu", "camera teruvu",
            "photo dekho", "image batao", "picture describe karo", "camera kholo",
            "kya dikh raha hai", "kya dikh rha hai",
        ]

        self.ocr_read_phrases = [
            "read this", "what is written", "read this bill", "read this receipt",
            "read this document", "read this invoice", "read this prescription",
            "iss bill me kya likha hai", "is bill me kya likha hai", "bill padho",
            "invoice read karo", "receipt batao", "document padho", "paper me kya likha hai",
            "qr code padho", "barcode read karo", "aadhaar", "pan card", "passport", "cheque",
            "पत्र पढ़ो", "इस बिल में क्या लिखा है", "डॉक्यूमेंट पढ़ो", "रसीद पढ़ो",
            "ఇది చదువు", "ఏమి రాసి ఉంది", "బిల్లు చదువు", "డాక్యుమెంట్ చదువు",
            "idi chaduvu", "em rasundi", "bill chaduvu", "document chaduvu",
        ]

        self.system_phrases = [
            "open camera", "close camera", "switch to hindi", "switch to english",
            "increase volume", "decrease volume", "stop speaking", "exit", "quit",
            "shutdown", "close app", "अब हिंदी में बात करो", "english mein baat karo",
        ]

        self.vision_nouns = {
            "camera", "image", "photo", "picture", "scene", "surroundings", "तस्वीर", "फोटो",
            "కెమెరా", "చిత్రం", "ఫోటో",
        }
        self.ocr_nouns = {
            "bill", "invoice", "receipt", "document", "prescription", "id", "paper", "form",
            "passport", "aadhaar", "pan", "cheque", "qr", "barcode", "newspaper", "letter",
            "बिल", "रसीद", "डॉक्यूमेंट", "कागज", "पत्र",
            "బిల్లు", "డాక్యుమెంట్", "కాగితం",
        }
        self.read_verbs = {
            "read", "written", "text", "padho", "padhna", "likha", "लिखा", "पढ़ो",
            "chaduvu", "rasundi", "చదువు", "రాసి",
        }
        self.see_verbs = {
            "see", "look", "describe", "show", "dekh", "dikha", "dikh", "दिख", "देख",
            "choodu", "choosi", "kanipistundi", "చూడు", "చూసి", "కనిపిస్తోంది",
        }
        self.vision_context_tokens = {"kya", "raha", "rha", "hai", "है", "क्या", "रहा"}

    def _normalize(self, text):
        lowered = text.lower().strip()
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered

    def _tokens(self, text):
        return re.findall(r"[a-zA-Z]+|[\u0900-\u097f]+|[\u0c00-\u0c7f]+", text)

    def _phrase_hits(self, text, phrase_list):
        return sum(1 for p in phrase_list if p in text)

    def route(self, text: str):
        query = self._normalize(text)
        tokens = set(self._tokens(query))

        system_hits = self._phrase_hits(query, self.system_phrases)
        ocr_phrase_hits = self._phrase_hits(query, self.ocr_read_phrases)
        vision_phrase_hits = self._phrase_hits(query, self.vision_action_phrases)

        ocr_noun_hits = len(tokens.intersection(self.ocr_nouns))
        vision_noun_hits = len(tokens.intersection(self.vision_nouns))
        read_hits = len(tokens.intersection(self.read_verbs))
        see_hits = len(tokens.intersection(self.see_verbs))
        vision_context_hits = len(tokens.intersection(self.vision_context_tokens))

        # Weighted semantic intent scoring (not single-keyword routing).
        scores = {
            "SYSTEM": 0.75 * system_hits,
            "OCR": 0.45 * ocr_phrase_hits + 0.30 * ocr_noun_hits + 0.25 * read_hits,
            "VISION": 0.50 * vision_phrase_hits + 0.30 * see_hits + 0.20 * vision_noun_hits,
            "CHAT": 0.15,
        }

        # Boost colloquial vision prompts like 'kya dikh raha hai'.
        if see_hits > 0 and vision_context_hits >= 2:
            scores["VISION"] += 0.35

        # A lone conversational word such as "see" in "see you later" is
        # not a camera request. Require an action phrase, image noun, or the
        # colloquial question context before selecting vision mode.
        has_vision_evidence = (
            vision_phrase_hits > 0
            or vision_noun_hits > 0
            or (see_hits > 0 and vision_context_hits >= 2)
        )
        if not has_vision_evidence:
            scores["VISION"] = 0

        # OCR should dominate when reading intent is present.
        if read_hits > 0 and (ocr_noun_hits > 0 or ocr_phrase_hits > 0):
            scores["OCR"] += 0.35

        # Penalize vision if question is clearly text-reading oriented.
        if scores["OCR"] > 0.4:
            scores["VISION"] *= 0.6

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        intent, top_score = ranked[0]
        second_score = ranked[1][1]

        # Confidence from margin + absolute signal strength.
        margin = max(0.0, top_score - second_score)
        base = max(top_score, 1e-6)
        confidence = min(0.99, max(0.35, 0.55 + 0.35 * (margin / base)))

        return {
            "intent": intent,
            "mode": intent,
            "confidence": round(confidence, 2),
            "scores": scores,
            "reason": f"top={intent}, margin={margin:.2f}",
        }