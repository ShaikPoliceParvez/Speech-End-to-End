import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest

from core.language import detect_dominant_language, detect_script, normalize_text


class LanguageDetectionTests(unittest.TestCase):
    def test_expected_language_and_script_examples(self):
        examples = {
            "Ni pairu enti": ("te", "latin"),
            "naku horror katha cheppu": ("te", "latin"),
            "malai euta katha sunau": ("ne", "latin"),
            "wow": ("en", "latin"),
            "thanks": ("en", "latin"),
            "that was fantastic": ("en", "latin"),
            "నాకు కథ చెప్పు": ("te", "telugu"),
            "मुझे कहानी सुनाओ": ("hi", "devanagari"),
            "म नेपालीमा बोल्छु": ("ne", "devanagari"),
            "Hello Tarz": ("en", "latin"),
            "good morning": ("en", "latin"),
        }

        for text, (expected_language, expected_script) in examples.items():
            with self.subTest(text=text):
                self.assertEqual(detect_dominant_language(text), expected_language)
                self.assertEqual(detect_script(text), expected_script)

    def test_previous_language_is_only_used_for_ambiguous_acknowledgements(self):
        self.assertEqual(detect_dominant_language("wow", previous_language="te"), "en")
        self.assertEqual(detect_dominant_language("ok", previous_language="te"), "te")

    def test_conflicting_roman_evidence_beats_whisper_hint(self):
        self.assertEqual(
            detect_dominant_language("naku katha cheppu", stt_hint="en", stt_confidence=0.95),
            "te",
        )

    def test_common_roman_hindi_trip_request(self):
        text = "mereliye 5 days bombay trip plan karo"
        self.assertEqual(detect_dominant_language(text), "hi")
        self.assertEqual(normalize_text(text, "hi"), "मेरे लिए 5 days bombay trip plan करो")

    def test_roman_hindi_majority_beats_shared_words(self):
        self.assertEqual(detect_dominant_language("tumhe bahubali ka story patahe"), "hi")

    def test_roman_hindi_feedback_not_stuck_in_previous_telugu(self):
        self.assertEqual(
            detect_dominant_language("already mene sunliya na", previous_language="te"),
            "hi",
        )

    def test_roman_telugu_vocabulary(self):
        self.assertEqual(detect_dominant_language("atanu ela chanipoyadu"), "te")

    def test_extended_roman_vocabulary(self):
        self.assertEqual(detect_dominant_language("can you explain this"), "en")
        self.assertEqual(detect_dominant_language("tum mujhe kya bataoge"), "hi")
        self.assertEqual(detect_dominant_language("tapai lai ke chha"), "ne")
        self.assertEqual(detect_dominant_language("nuvvu naaku cheptava"), "te")
        self.assertEqual(detect_dominant_language("nuv naku sahayam chestava"), "te")

    def test_explicit_switch_to_nepali(self):
        self.assertEqual(detect_dominant_language("please switch language to nepali"), "ne")

    def test_hindi_devanagari_not_misclassified_as_nepali(self):
        # Even with a conflicting high-confidence hint, Hindi Devanagari should stay Hindi.
        self.assertEqual(
            detect_dominant_language("मुझे कहानी सुनाओ", stt_hint="ne", stt_confidence=0.95),
            "hi",
        )

    def test_ambiguous_devanagari_defaults_to_hindi(self):
        # Shared Devanagari words should not randomly switch to Nepali.
        self.assertEqual(detect_dominant_language("धन्यवाद"), "hi")

    def test_arabic_and_mixed_scripts_are_reported(self):
        self.assertEqual(detect_script("سلام"), "arabic")
        self.assertEqual(detect_script("తెలుగు hello"), "mixed")


if __name__ == "__main__":
    unittest.main()