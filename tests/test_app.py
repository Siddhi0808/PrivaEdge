"""Tests for UIChat.py and TerminalChat.py.

Run from the project root:  python -m unittest discover -s tests -v

Tests marked "needs Ollama" talk to the local Ollama server (http://localhost:11434)
and are skipped automatically when it is not running.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.request

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_APP = os.path.join(ROOT, "UIChat.py")
TERMINAL_APP = os.path.join(ROOT, "TerminalChat.py")


def ollama_models():
    """Return installed Ollama model names, or None if the server is not reachable."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
            return [m["name"] for m in json.load(resp)["models"]]
    except Exception:
        return None


OLLAMA = ollama_models()
needs_ollama = unittest.skipIf(OLLAMA is None, "Ollama server not running")
needs_llama = unittest.skipIf(
    not OLLAMA or not any(m.startswith("llama3.2") for m in OLLAMA), "llama3.2 not installed"
)


class UIChatTests(unittest.TestCase):
    def setUp(self):
        # UIChat writes sessions to ./chat_sessions, so run each test in a scratch directory
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def run_app(self):
        at = AppTest.from_file(UI_APP, default_timeout=180)
        at.run()
        self.assertFalse(at.exception, f"app raised: {at.exception}")
        return at

    def test_app_renders_with_chat_input(self):
        at = self.run_app()
        self.assertEqual(len(at.chat_input), 1)
        self.assertEqual(at.session_state.session_title, "New Chat")

    @needs_ollama
    def test_model_selector_excludes_embedding_only_models(self):
        at = self.run_app()
        options = at.selectbox[0].options
        self.assertTrue(options)
        for name in options:
            self.assertNotIn("embed", name, f"embedding model offered for chat: {name}")

    @needs_llama
    def test_chat_round_trip_saves_session(self):
        at = self.run_app()
        at.selectbox[0].select(next(m for m in at.selectbox[0].options if m.startswith("llama3.2")))
        at.chat_input[0].set_value("Reply with the single word: pong").run()
        self.assertFalse(at.exception, f"app raised: {at.exception}")

        messages = at.session_state.messages
        self.assertEqual([m.type for m in messages], ["human", "ai"])
        self.assertTrue(messages[1].content.strip())
        self.assertNotIn("technical difficulties", messages[1].content)

        files = os.listdir("chat_sessions")
        self.assertEqual(len(files), 1)
        with open(os.path.join("chat_sessions", files[0])) as f:
            saved = json.load(f)
        self.assertEqual(saved["title"], "Reply with the single...")
        self.assertEqual([m["type"] for m in saved["messages"]], ["human", "ai"])

    def test_llm_failure_shows_fallback_without_storing_it(self):
        # Point the Ollama client at a closed port to simulate the server being down.
        # (ChatOllama builds a new client per call and reads OLLAMA_HOST each time.)
        old_host = os.environ.get("OLLAMA_HOST")
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"
        try:
            at = self.run_app()
            at.chat_input[0].set_value("hello").run()
        finally:
            if old_host is None:
                os.environ.pop("OLLAMA_HOST", None)
            else:
                os.environ["OLLAMA_HOST"] = old_host
        self.assertFalse(at.exception, f"app raised: {at.exception}")
        self.assertTrue(at.error, "expected an error message")
        self.assertIn("technical difficulties", at.chat_message[-1].markdown[0].value)
        # The canned reply must not become part of the conversation sent back to the model
        self.assertEqual([m.type for m in at.session_state.messages], ["human"])

    def test_saved_session_appears_in_sidebar_and_reloads(self):
        os.makedirs("chat_sessions", exist_ok=True)
        with open(os.path.join("chat_sessions", "abc.json"), "w") as f:
            json.dump({"title": "Old chat", "timestamp": "2026-01-01T10:00:00",
                       "messages": [{"type": "human", "content": "hi"},
                                    {"type": "ai", "content": "hello!"}]}, f)
        at = self.run_app()
        session_button = next(b for b in at.button if b.label == "💬 Old chat")
        session_button.click().run()
        self.assertEqual(at.session_state.current_session_id, "abc")
        self.assertEqual([m.content for m in at.session_state.messages], ["hi", "hello!"])

    def test_corrupt_session_file_is_skipped(self):
        os.makedirs("chat_sessions", exist_ok=True)
        with open(os.path.join("chat_sessions", "broken.json"), "w") as f:
            f.write("{not valid json")
        at = self.run_app()
        self.assertFalse(any(b.label.startswith("💬 ") and "broken" in b.label for b in at.button))


WHISPER_WEIGHTS = os.path.join(ROOT, "models", "base.pt")
HAS_SAY = sys.platform == "darwin" and os.path.exists("/usr/bin/say")


class VoiceWorkerTests(unittest.TestCase):
    """The speech workers run as separate processes (see whisper_transcribe.py / tts_worker.py)."""

    def test_tts_worker_writes_non_empty_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "speech")
            subprocess.run([sys.executable, os.path.join(ROOT, "tts_worker.py"), "save", out],
                           input="Testing offline speech synthesis.", text=True, check=True, timeout=60)
            # An empty header-only file (the old in-thread bug) is about 4 KB; real speech is far larger
            self.assertGreater(os.path.getsize(out), 20_000)

    @unittest.skipUnless(HAS_SAY and os.path.exists(WHISPER_WEIGHTS),
                         "needs macOS `say` and cached Whisper weights in models/")
    def test_whisper_worker_transcribes_spoken_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = os.path.join(tmp, "clip.aiff")
            subprocess.run(["say", "-o", clip, "What is the capital city of Japan?"], check=True)
            result = subprocess.run([sys.executable, os.path.join(ROOT, "whisper_transcribe.py"), clip],
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr[-500:])
            text = json.loads(result.stdout.strip().splitlines()[-1])["text"].lower()
            self.assertIn("capital", text)
            self.assertIn("japan", text)

    def test_whisper_worker_reports_missing_file(self):
        result = subprocess.run([sys.executable, os.path.join(ROOT, "whisper_transcribe.py"),
                                 "/nonexistent/audio.mp3"], capture_output=True, text=True, timeout=180)
        self.assertNotEqual(result.returncode, 0)


class TerminalChatTests(unittest.TestCase):
    def run_cli(self, stdin_text):
        return subprocess.run([sys.executable, TERMINAL_APP], input=stdin_text,
                              capture_output=True, text=True, timeout=180)

    def test_exits_cleanly_on_end_of_input(self):
        result = self.run_cli("")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_blank_lines_are_ignored_and_exit_works(self):
        result = self.run_cli("\n   \nexit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Bot:", result.stdout)

    @needs_llama
    def test_answers_and_remembers_history(self):
        result = self.run_cli("My name is Ada. Reply OK.\nWhat is my name? Answer in one word.\nquit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        # With piped stdin the "You: " prompt and the reply share a line, e.g. "You: Bot: OK"
        replies = [line.split("Bot:", 1)[1] for line in result.stdout.splitlines() if "Bot:" in line]
        self.assertEqual(len(replies), 2)
        self.assertIn("ada", replies[1].lower())


if __name__ == "__main__":
    unittest.main()
