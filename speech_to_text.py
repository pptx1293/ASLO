"""
ASLO Speech-To-Text (STT) Engine
Provides robust speech recognition capabilities from browser-recorded WAV streams,
system microphones, and audio files using Google Speech Recognition API and sounddevice.
"""

import io
import os
import sys
import time
import argparse
import numpy as np

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    wavfile = None

SUPPORTED_LANGUAGES = {
    "en-US": "English (US)",
    "vi-VN": "Vietnamese (Tiếng Việt)",
    "es-ES": "Spanish (Español)",
    "fr-FR": "French (Français)",
    "de-DE": "German (Deutsch)",
    "ja-JP": "Japanese (日本語)",
    "ko-KR": "Korean (한국어)",
    "zh-CN": "Chinese (Mandarin)"
}


class SpeechToTextEngine:
    def __init__(self):
        if sr is not None:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
        else:
            self.recognizer = None

    def transcribe_wav_bytes(self, wav_bytes: bytes, language: str = "en-US") -> dict:
        """Transcribe raw WAV bytes in-memory without creating temporary files on disk."""
        if self.recognizer is None:
            return {
                "ok": False,
                "error": "SpeechRecognition library is not available. Please run: pip install SpeechRecognition"
            }

        if not wav_bytes or len(wav_bytes) < 44:
            return {"ok": False, "error": "Invalid or empty audio data received."}

        try:
            buf = io.BytesIO(wav_bytes)
            with sr.AudioFile(buf) as source:
                audio = self.recognizer.record(source)

            if len(audio.get_raw_data()) == 0:
                return {"ok": False, "error": "Audio stream contains no recorded data."}

            text = self.recognizer.recognize_google(audio, language=language)
            return {
                "ok": True,
                "text": text.strip(),
                "language": language
            }
        except sr.UnknownValueError:
            return {
                "ok": False,
                "error": "Could not understand audio. Please speak more clearly or adjust microphone volume.",
                "reason": "unknown_speech"
            }
        except sr.RequestError as e:
            return {
                "ok": False,
                "error": f"Speech Recognition service error: {e}",
                "reason": "request_failed"
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"Failed to process audio: {str(e)}",
                "reason": "processing_error"
            }

    def transcribe_file(self, file_path: str, language: str = "en-US") -> dict:
        """Transcribe an audio file from the filesystem."""
        if not os.path.exists(file_path):
            return {"ok": False, "error": f"File '{file_path}' not found."}

        try:
            with open(file_path, "rb") as f:
                data = f.read()
            return self.transcribe_wav_bytes(data, language=language)
        except Exception as e:
            return {"ok": False, "error": f"Error reading file '{file_path}': {e}"}

    def record_system_microphone(self, duration: float = 4.0, sample_rate: int = 16000, language: str = "en-US") -> dict:
        """Record audio directly from the host system microphone using sounddevice."""
        try:
            import sounddevice as sd
        except ImportError:
            return {"ok": False, "error": "sounddevice library is not installed."}

        if wavfile is None:
            return {"ok": False, "error": "scipy is not installed for audio processing."}

        try:
            num_samples = int(duration * sample_rate)
            recording = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()

            buf = io.BytesIO()
            wavfile.write(buf, sample_rate, recording)
            wav_bytes = buf.getvalue()

            return self.transcribe_wav_bytes(wav_bytes, language=language)
        except Exception as e:
            return {"ok": False, "error": f"Microphone recording failed: {e}"}


# Global singleton instance for easy import
engine = SpeechToTextEngine()


def main():
    parser = argparse.ArgumentParser(description="ASLO Speech-To-Text Utility")
    parser.add_argument("--record", type=float, help="Record audio from PC microphone for N seconds and transcribe.")
    parser.add_argument("--file", type=str, help="Transcribe audio file path.")
    parser.add_argument("--lang", type=str, default="en-US", help="Language code (e.g., en-US, vi-VN, es-ES).")
    parser.add_argument("--test", action="store_true", help="Run self-test with synthetic audio.")
    parser.add_argument("--list-devices", action="store_true", help="List audio input/output devices.")
    args = parser.parse_args()

    stt = SpeechToTextEngine()

    if args.list_devices:
        try:
            import sounddevice as sd
            print("Available audio devices:")
            print(sd.query_devices())
        except Exception as e:
            print(f"Error listing devices: {e}")
        return

    if args.test:
        print("Testing Speech-To-Text Engine with synthesized speech...")
        try:
            import pyttsx3
            tts = pyttsx3.init()
            test_file = "temp_stt_test.wav"
            tts.save_to_file("Hello world how are you today", test_file)
            tts.runAndWait()

            res = stt.transcribe_file(test_file, language=args.lang)
            if os.path.exists(test_file):
                os.remove(test_file)

            print("Test Result:", res)
            if res.get("ok"):
                print("Self-test PASSED successfully!")
            else:
                print("Self-test completed with notification:", res.get("error"))
        except Exception as e:
            print("Self-test error:", e)
        return

    if args.file:
        print(f"Transcribing file: {args.file} (Language: {args.lang})...")
        res = stt.transcribe_file(args.file, language=args.lang)
        print("Result:", res)
        return

    if args.record:
        print(f"Recording from microphone for {args.record}s (Language: {args.lang})... Please speak now.")
        res = stt.record_system_microphone(duration=args.record, language=args.lang)
        print("Result:", res)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
