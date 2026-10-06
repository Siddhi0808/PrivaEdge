"""Offline text-to-speech worker using pyttsx3 (the operating system's built-in voices).

UIChat.py runs this script as a separate process. Streamlit executes the app in a worker
thread, and the macOS speech driver only produces audio when it runs on a process's main
thread (from a worker thread it writes an empty file and speaks nothing). A short-lived
process gives pyttsx3 its own main thread.

Usage (the text to speak is read from stdin):
    python tts_worker.py speak               # speak aloud through the speakers
    python tts_worker.py save <output_path>  # write the speech to an audio file
"""
import sys

import pyttsx3


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("speak", "save") or (sys.argv[1] == "save" and len(sys.argv) < 3):
        sys.exit("usage: python tts_worker.py speak | save <output_path>  (text on stdin)")
    text = sys.stdin.read()  # text arrives on stdin, so long answers need no command-line escaping

    engine = pyttsx3.init()
    engine.setProperty('rate', 175)    # words per minute
    engine.setProperty('volume', 0.9)  # 0.0 - 1.0
    if sys.argv[1] == "speak":
        engine.say(text)
    else:
        engine.save_to_file(text, sys.argv[2])
    engine.runAndWait()  # blocks until speaking / writing the file has finished


if __name__ == "__main__":
    main()
