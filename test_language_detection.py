import unittest

from language import detect_dominant_language, detect_script


class LanguageDetectionTests(unittest.TestCase):
    def test_expected_language_and_script_examples(self):
        examples = {
            "Ni pairu enti": ("te", "latin"),
            "naku horror katha cheppu": ("te", "latin"),
            "wow": ("en", "latin"),
            "thanks": ("en", "latin"),
            "that was fantastic": ("en", "latin"),
            "నాకు కథ చెప్పు": ("te", "telugu"),
            "मुझे कहानी सुनाओ": ("hi", "devanagari"),
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

    def test_arabic_and_mixed_scripts_are_reported(self):
        self.assertEqual(detect_script("سلام"), "arabic")
        self.assertEqual(detect_script("తెలుగు hello"), "mixed")


if __name__ == "__main__":
    unittest.main()