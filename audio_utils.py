"""
Audio Utility Module - High-level interface for recording and transcription.
Provides smart, efficient, and easy-to-use functions for audio processing.
"""

from microphone import Microphone
from stt import STT
import numpy as np
from config import SAMPLE_RATE


class AudioProcessor:
    """
    Unified interface for recording and transcription with smart defaults.
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.mic = Microphone(silence_threshold=0.01, silence_duration=1.5)
        self.stt = STT(verbose=verbose)
    
    def record_and_transcribe(self, output_file="recording.wav", auto_stop=True):
        """
        Record audio and transcribe it in one call.
        
        Args:
            output_file: Where to save the audio
            auto_stop: Auto-stop on silence (smarter than manual)
            
        Returns:
            dict with audio file, transcription, language, and confidence
        """
        print("\n" + "="*60)
        print("RECORD & TRANSCRIBE")
        print("="*60)
        
        result = self.mic.record_with_realtime_transcription(
            self.stt,
            filename=output_file,
            verbose=self.verbose
        )
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Audio: {result['audio_file']}")
        print(f"Transcription: {result['final_text']}")
        print(f"Language: {result['language']}")
        print(f"Real-time updates: {len(result['transcriptions'])}")
        
        return result
    
    def quick_transcribe(self, audio_file):
        """
        Quickly transcribe an existing audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text
        """
        return self.stt.transcribe(audio_file, return_meta=False)
    
    def transcribe_with_details(self, audio_file):
        """
        Transcribe with detailed metadata.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            dict with text, language, confidence, and segments
        """
        return self.stt.transcribe(audio_file, return_meta=True, return_segments=True)
    
    def cleanup(self):
        """Clean up resources"""
        self.stt.cleanup_temp_files()


def record_and_transcribe_simple(output_file="recording.wav", auto_stop=True):
    """
    Simple one-liner for record and transcribe.
    
    Usage:
        result = record_and_transcribe_simple()
        print(result['final_text'])
    """
    processor = AudioProcessor(verbose=False)
    result = processor.record_and_transcribe(output_file, auto_stop)
    processor.cleanup()
    return result


def quick_transcribe(audio_file):
    """
    Simple one-liner to transcribe a file.
    
    Usage:
        text = quick_transcribe("speech.wav")
        print(text)
    """
    stt = STT(verbose=False)
    text = stt.transcribe(audio_file)
    stt.cleanup_temp_files()
    return text


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AUDIO PROCESSING UTILITY")
    print("="*60)
    
    print("\n1. Record and Transcribe")
    print("2. Transcribe Existing File")
    print("3. Full Integration Test")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        result = record_and_transcribe_simple(auto_stop=True)
        print(f"\n✓ Final Text: {result['final_text']}")
        print(f"✓ Language: {result['language']}")
    
    elif choice == "2":
        file = input("Enter audio file path: ").strip()
        if file:
            text = quick_transcribe(file)
            print(f"\n✓ Transcription: {text}")
    
    elif choice == "3":
        print("\nRunning full integration test...")
        processor = AudioProcessor(verbose=True)
        result = processor.record_and_transcribe(auto_stop=True)
        details = processor.transcribe_with_details(result['audio_file'])
        
        print("\n" + "="*60)
        print("SEGMENT ANALYSIS")
        print("="*60)
        for i, seg in enumerate(details.get('segments', []), 1):
            print(f"Segment {i}: {seg['text']}")
            print(f"  Time: {seg['start']:.2f}s - {seg['end']:.2f}s")
            print(f"  Confidence: {seg.get('confidence', 'N/A')}")
        
        processor.cleanup()
    
    else:
        print("Invalid option")
