import re


class Router:

    def __init__(self):
        # Semantic phrase banks (multilingual), scored by intent groups.
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
            "switch to hindi", "switch to english",
            "increase volume", "decrease volume", "stop speaking", "exit", "quit",
            "shutdown", "close app", "अब हिंदी में बात करो", "english mein baat karo",
        ]

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

        ocr_noun_hits = len(tokens.intersection(self.ocr_nouns))
        read_hits = len(tokens.intersection(self.read_verbs))

        # Weighted semantic intent scoring (not single-keyword routing).
        scores = {
            "SYSTEM": 0.75 * system_hits,
            "OCR": 0.45 * ocr_phrase_hits + 0.30 * ocr_noun_hits + 0.25 * read_hits,
            "CHAT": 0.15,
        }

        # OCR should dominate when reading intent is present.
        if read_hits > 0 and (ocr_noun_hits > 0 or ocr_phrase_hits > 0):
            scores["OCR"] += 0.35

        # Explicitly combined requests should preserve both capabilities.
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