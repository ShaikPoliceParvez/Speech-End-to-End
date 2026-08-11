import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STT_ALLOWED_LANGUAGES, SUPPORTED_LANGUAGES, TTS_LANGUAGE_BACKENDS
from core.language import detect_dominant_language, detect_script


def _check(condition, label, failures):
	if condition:
		print(f"[PASS] {label}")
	else:
		print(f"[FAIL] {label}")
		failures.append(label)


def _validate_language_detection(failures):
	print("\n== Language Detection ==")
	cases = [
		("hello, how are you", "en"),
		("मुझे कहानी सुनाओ", "hi"),
		("malai euta katha sunau", "ne"),
		("म नेपालीमा बोल्छु", "ne"),
		("నాకు కథ చెప్పు", "te"),
		("naku katha cheppu", "te"),
		("എനിക്ക് ഒരു കഥ പറയൂ", "ml"),
		("njan oru katha parayu", "ml"),
		("مرحبا كيف حالك", "ar"),
		("marhaba kaif halak", "ar"),
	]

	for text, expected in cases:
		actual = detect_dominant_language(text)
		_check(
			actual == expected,
			f"detect_dominant_language('{text[:24]}...') -> {expected} (got {actual})",
			failures,
		)

	# Ensure strong Roman evidence beats conflicting stale hints.
	hint_case = detect_dominant_language(
		"naku katha cheppu", stt_hint="en", stt_confidence=0.95
	)
	_check(
		hint_case == "te",
		"Roman evidence overrides conflicting high-confidence STT hint",
		failures,
	)

	_check(detect_script("म नेपालीमा बोल्छु") == "devanagari", "Nepali Devanagari script detection", failures)
	_check(detect_script("తెలుగు hello") == "mixed", "Mixed script detection", failures)


def _validate_pipeline_maps(root_dir, strict_models, failures):
	print("\n== Pipeline Maps ==")
	supported = set(SUPPORTED_LANGUAGES.keys())
	stt_supported = set(STT_ALLOWED_LANGUAGES)
	tts_mapped = set(TTS_LANGUAGE_BACKENDS.keys())

	_check(supported.issubset(stt_supported), "All UI languages are present in STT_ALLOWED_LANGUAGES", failures)
	_check(supported.issubset(tts_mapped), "All UI languages have TTS backend mapping", failures)

	for lang in sorted(supported):
		route = TTS_LANGUAGE_BACKENDS.get(lang)
		if not route:
			continue

		backend = route.get("backend")
		_check(backend in {"supertonic", "piper"}, f"TTS backend valid for '{lang}'", failures)

		if backend == "piper":
			model_rel = route.get("model")
			model_abs = root_dir / model_rel
			model_json = Path(str(model_abs) + ".json")
			model_ok = model_abs.is_file()
			json_ok = model_json.is_file()

			if strict_models:
				_check(model_ok, f"Piper ONNX exists for '{lang}' -> {model_rel}", failures)
				_check(json_ok, f"Piper metadata exists for '{lang}' -> {model_rel}.json", failures)
			else:
				# Non-strict mode still reports but does not fail hard for missing model metadata.
				status = "PASS" if model_ok and json_ok else "WARN"
				print(
					f"[{status}] Piper files for '{lang}': "
					f"onnx={'ok' if model_ok else 'missing'}, json={'ok' if json_ok else 'missing'}"
				)
				if not model_ok:
					failures.append(f"Missing Piper ONNX for '{lang}'")


def main():
	parser = argparse.ArgumentParser(description="Tarz multilingual pipeline smoke test")
	parser.add_argument(
		"--strict-models",
		action="store_true",
		help="Fail when any Piper .onnx.json metadata file is missing",
	)
	args = parser.parse_args()

	root_dir = Path(__file__).resolve().parent.parent
	failures = []

	print("Tarz Smoke Check")
	print(f"Workspace: {root_dir}")
	print(f"Languages: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}")

	_validate_language_detection(failures)
	_validate_pipeline_maps(root_dir, args.strict_models, failures)

	print("\n== Result ==")
	if failures:
		print(f"FAIL ({len(failures)} issue(s))")
		for item in failures:
			print(f" - {item}")
		raise SystemExit(1)

	print("PASS")


if __name__ == "__main__":
	main()
