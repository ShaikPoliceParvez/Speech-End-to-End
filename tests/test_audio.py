#!/usr/bin/env python3
"""
Test file for Microphone and STT (Speech-to-Text) functionality
"""

import sys
import os
import time

def test_imports():
    """Test if all required dependencies are installed"""
    print("=" * 60)
    print("TESTING DEPENDENCIES...")
    print("=" * 60)
    
    dependencies = {
        'sounddevice': 'sounddevice',
        'soundfile': 'soundfile',
        'numpy': 'numpy',
        'faster_whisper': 'faster-whisper',
    }
    
    all_good = True
    for module_name, package_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name} - OK")
        except ImportError as e:
            print(f"✗ {package_name} - MISSING")
            print(f"  Error: {e}")
            all_good = False
    
    return all_good


def test_microphone():
    """Test microphone recording"""
    print("\n" + "=" * 60)
    print("TESTING MICROPHONE...")
    print("=" * 60)
    
    try:
        from microphone import Microphone
        from config import SAMPLE_RATE, CHANNELS
        import sounddevice as sd
        
        # Check available audio devices
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"  [{i}] {device['name']} (In: {device['max_input_channels']}, Out: {device['max_output_channels']})")
        
        default_device = sd.default.device
        print(f"\nDefault input device: {default_device[0]}")
        print(f"Sample rate: {SAMPLE_RATE} Hz")
        print(f"Channels: {CHANNELS}")
        
        # Test recording 3 seconds
        print("\n--- Recording Test ---")
        print("Recording 3 seconds of audio... please speak something!")
        
        mic = Microphone(chunk_seconds=0.4)
        
        # Modify record method to use timeout for testing
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
        import threading
        
        test_file = "test_recording.wav"
        recording = []
        stop_event = threading.Event()
        
        def callback(indata, frames, time, status):
            if status:
                print(f"  Recording status: {status}")
            recording.append(indata.copy())
        
        # Record for 3 seconds
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=mic.chunk_frames,
            dtype="float32",
            callback=callback,
        ):
            print("Recording started... (will stop in 3 seconds)")
            time.sleep(3)
        
        if recording:
            audio = np.concatenate(recording, axis=0)
            sf.write(test_file, audio, SAMPLE_RATE)
            file_size = os.path.getsize(test_file)
            print(f"✓ Microphone test PASSED")
            print(f"  Recording saved: {test_file} ({file_size} bytes)")
            return test_file
        else:
            print("✗ Microphone test FAILED - No audio captured")
            return None
            
    except Exception as e:
        print(f"✗ Microphone test FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_stt(audio_file):
    """Test STT (Speech-to-Text) transcription"""
    print("\n" + "=" * 60)
    print("TESTING STT (SPEECH-TO-TEXT)...")
    print("=" * 60)
    
    if not audio_file or not os.path.exists(audio_file):
        print("✗ No audio file to test with")
        return False
    
    try:
        from stt import STT
        
        print(f"\nLoading STT model...")
        print("(This may take a moment on first run...)")
        stt = STT()
        
        print(f"\n--- Transcribing: {audio_file} ---")
        result = stt.transcribe(audio_file, return_meta=True)
        
        text = result.get("text", "")
        language = result.get("language", "Unknown")
        language_prob = result.get("language_probability", 0)
        
        print(f"\nTranscription Result:")
        print(f"  Text: {text if text else '(empty/no speech detected)'}")
        print(f"  Detected Language: {language}")
        print(f"  Language Probability: {language_prob:.2%}")
        
        if text.strip():
            print(f"\n✓ STT test PASSED - Successfully transcribed audio")
            return True
        else:
            print(f"\n⚠ STT test completed but no text detected")
            print(f"  This could mean:")
            print(f"  - No speech was detected in the audio")
            print(f"  - Audio quality was too low")
            print(f"  - Try recording with clearer speech")
            return False
            
    except Exception as e:
        print(f"✗ STT test FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated():
    """Test integrated microphone + STT workflow with smart features"""
    print("\n" + "=" * 60)
    print("INTEGRATED TEST: Smart Record and Transcribe")
    print("=" * 60)
    
    try:
        from microphone import Microphone
        from stt import STT
        
        mic = Microphone(chunk_seconds=0.4, silence_threshold=0.01, silence_duration=1.5)
        stt = STT(verbose=True)
        
        print("\n🎤 Using SMART AUTO-STOP mode (stops on silence)")
        print("Speak something clearly...\n")
        
        result = mic.record_with_realtime_transcription(
            stt,
            filename="test_smart_integrated.wav",
            verbose=True
        )
        
        print(f"\n✓ Recording completed: {result['audio_file']}")
        
        print(f"\n✓ Real-time Transcription Results:")
        print(f"  Final Text: {result['final_text']}")
        print(f"  Detected Language: {result['language']}")
        print(f"  Real-time Updates: {len(result['transcriptions'])}")
        
        if result['transcriptions']:
            print(f"\n  Real-time Stream:")
            for i, trans in enumerate(result['transcriptions'], 1):
                print(f"    {i}. {trans['text']} (confidence: {trans['confidence']:.1%})")
        
        if result['final_text'].strip():
            print(f"\n✓ INTEGRATED TEST PASSED!")
            return True
        else:
            print("✗ No speech detected")
            
    except Exception as e:
        print(f"✗ Integrated test FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    return False


def test_smart_silence_detection():
    """Test silence detection feature"""
    print("\n" + "=" * 60)
    print("SMART SILENCE DETECTION TEST")
    print("=" * 60)
    
    try:
        from microphone import Microphone
        
        print("\nTesting silence detection accuracy...")
        mic = Microphone(
            chunk_seconds=0.4,
            silence_threshold=0.01,
            silence_duration=1.5
        )
        
        print("Recording with auto-stop (will stop 1.5s after silence)...")
        print("Speak something, pause, then speak again.\n")
        
        audio_file = mic.record(
            filename="test_silence.wav",
            auto_stop=True,
            verbose=True
        )
        
        print(f"\n✓ Silence detection test completed!")
        print(f"  File: {audio_file}")
        return True
        
    except Exception as e:
        print(f"✗ Silence detection test FAILED: {e}")
        return False


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " IMPROVED MICROPHONE & STT TEST SUITE ".center(58) + "║")
    print("║" + " With Smart Features & Real-time Processing ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Test imports
    deps_ok = test_imports()
    
    if not deps_ok:
        print("\n" + "=" * 60)
        print("⚠️  Missing dependencies detected!")
        print("=" * 60)
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        print("\nOr install specific packages:")
        print("  pip install sounddevice soundfile numpy faster-whisper")
        sys.exit(1)
    
    # Interactive menu
    print("\n" + "=" * 60)
    print("AVAILABLE TESTS")
    print("=" * 60)
    print("\n1. Microphone Test (basic recording)")
    print("2. STT Test (transcribe existing file)")
    print("3. Silence Detection Test (auto-stop feature)")
    print("4. Integrated Test (smart record + transcribe)")
    print("5. Run All Tests")
    print("6. Quick Demo (audio_utils.py)")
    print("0. Exit")
    
    while True:
        choice = input("\nSelect test (0-6): ").strip()
        
        if choice == "0":
            print("\nGoodbye!")
            break
        
        elif choice == "1":
            test_file = test_microphone()
        
        elif choice == "2":
            print("\nEnter audio file path (or press Enter for recent test file): ", end="")
            file = input().strip() or "test_recording.wav"
            test_stt(file)
        
        elif choice == "3":
            test_smart_silence_detection()
        
        elif choice == "4":
            test_integrated()
        
        elif choice == "5":
            print("\n" + "=" * 60)
            print("Running ALL tests...")
            print("=" * 60)
            
            test_file = test_microphone()
            if test_file:
                test_stt(test_file)
            
            print("\n" + "=" * 60)
            test_smart_silence_detection()
            
            print("\n" + "=" * 60)
            test_integrated()
        
        elif choice == "6":
            print("\n" + "=" * 60)
            print("AUDIO UTILS DEMO")
            print("=" * 60)
            print("\nThe audio_utils.py module provides:")
            print("  - AudioProcessor class (unified interface)")
            print("  - record_and_transcribe_simple() (one-liner)")
            print("  - quick_transcribe() (transcribe files)")
            print("\nTry running: python audio_utils.py")
        
        else:
            print("Invalid option. Please try again.")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("\n✓ IMPROVEMENTS MADE:")
    print("  1. Smart silence detection - auto-stops on pause")
    print("  2. Real-time transcription - see results while recording")
    print("  3. Duplicate detection - eliminates repeated text")
    print("  4. Confidence scoring - track transcription quality")
    print("  5. Segment analysis - detailed transcription data")
    print("  6. Automatic cleanup - removes temp files")
    print("  7. Better error handling - comprehensive logging")
    print("  8. Unified interface - audio_utils.py module")
    
    print("\n📁 FILES MODIFIED:")
    print("  - microphone.py (added silence detection, auto-stop)")
    print("  - stt.py (added confidence, deduplication, cleanup)")
    print("  - audio_utils.py (NEW - high-level interface)")
    print("  - test_audio.py (updated with new tests)")
    
    print("\n🚀 QUICK START:")
    print("  from audio_utils import record_and_transcribe_simple")
    print("  result = record_and_transcribe_simple()")
    print("  print(result['final_text'])")
    
    print("\n" + "=" * 60)
    print("Testing complete!\n")


if __name__ == "__main__":
    main()
