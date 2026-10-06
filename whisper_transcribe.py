"""Offline speech-to-text worker: transcribes one audio file with the local Whisper model.

UIChat.py runs this script as a separate process instead of importing Whisper directly.
Whisper (via PyTorch) and FAISS each bundle their own OpenMP runtime (libomp), and loading
both into the same process aborts or deadlocks it on macOS. Keeping Whisper in a short-lived
child process means the two runtimes never meet.

Usage:  python whisper_transcribe.py <audio_path> [model_name]
Prints a JSON object {"text": "..."} to stdout.
"""
import json
import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python whisper_transcribe.py <audio_path> [model_name]")
    audio_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "base"

    import whisper  # imported here so the slow PyTorch import happens only in this worker process

    # Weights are downloaded into ./models on first use, then loaded from disk (offline) afterwards
    model = whisper.load_model(model_name, download_root=MODELS_DIR)
    result = model.transcribe(audio_path)  # ffmpeg decodes the audio; language is auto-detected
    # One JSON line on stdout is the result UIChat.py reads; warnings go to stderr
    print(json.dumps({"text": result.get("text", "").strip()}))


if __name__ == "__main__":
    main()
