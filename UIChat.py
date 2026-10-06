"""Offline AI Assistant - Streamlit web app.

Features: local LLM chat (Ollama), Q&A over an uploaded PDF/TXT (RAG with FAISS),
offline speech-to-text (Whisper), offline text-to-speech (pyttsx3) and saved chat history.

How Streamlit runs this file: the WHOLE script re-executes from top to bottom every time the
user interacts with the page (types, clicks, uploads). Ordinary variables are therefore reset
on each run; anything that must survive between runs lives in `st.session_state`, a
per-browser-tab dictionary (conversation, FAISS index, ids of the last processed files...).

Run with:  streamlit run UIChat.py
"""
import os
import sys
import json
import uuid
import tempfile
import subprocess
from datetime import datetime

import streamlit as st
import speech_recognition as sr

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import rag  # document Q&A pipeline: chunking, embeddings, FAISS retrieval, prompt building

# === Constants & Path Setup ===
HISTORY_DIR = "chat_sessions"  # one JSON file per conversation (relative to the working directory)
# Speech-to-text and text-to-speech run as separate worker processes (see run_whisper / speak)
WHISPER_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper_transcribe.py")
TTS_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_worker.py")
os.makedirs(HISTORY_DIR, exist_ok=True)

# === Custom CSS for Centered Spinner ===
# Shown in the middle of the screen while the model is generating an answer
spinner_css = """
<style>
.custom-loader {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 9999;
  width: 80px;
  height: 80px;
  border: 10px solid #f3f3f3;
  border-top: 10px solid #3498db;
  border-radius: 50%;
  animation: spin 1.2s linear infinite;
}
@keyframes spin {
  0% { transform: translate(-50%, -50%) rotate(0deg);}
  100% { transform: translate(-50%, -50%) rotate(360deg);}
}
</style>
"""

# === Utility Functions ===
def speak(text):
    """Speak the text aloud in a background process so the UI is not blocked.

    pyttsx3 runs in its own process (tts_worker.py) because the macOS speech driver
    produces no audio when called from Streamlit's script thread.
    """
    try:
        proc = subprocess.Popen([sys.executable, TTS_WORKER, "speak"], stdin=subprocess.PIPE, text=True)
        proc.stdin.write(text)
        proc.stdin.close()
    except Exception:
        pass

def text_to_speech_and_save(text):
    """Synthesize the text to a WAV file, then show an audio player and a download button.

    Runs the TTS worker synchronously (the player needs the finished file). Temporary files
    are always removed, and a failure only shows a warning - the chat itself is unaffected.
    """
    # Reserve a temp file name for the worker to write into
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tts") as tmp:
        raw_path = tmp.name
    wav_path = raw_path + ".wav"
    try:
        subprocess.run([sys.executable, TTS_WORKER, "save", raw_path], input=text, text=True,
                       check=True, capture_output=True)
        with open(raw_path, "rb") as f:
            audio_bytes = f.read()
        # The macOS speech driver writes AIFF data regardless of the file name; convert it to real WAV
        if not audio_bytes.startswith(b"RIFF"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_path, wav_path],
                           check=True, capture_output=True)
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")
        st.download_button("Download Audio", audio_bytes, file_name="response.wav", mime="audio/wav")
    except Exception as e:
        st.warning(f"⚠️ Could not generate audio file: {e}")
    finally:
        for path in (raw_path, wav_path):
            if os.path.exists(path):
                os.remove(path)

def run_whisper(audio_path, model_name="base"):
    """Transcribe an audio file offline with Whisper in a separate process.

    Whisper (PyTorch) and FAISS bundle conflicting OpenMP runtimes that abort or deadlock
    a process that loads both, so Whisper never runs inside the Streamlit process.
    See whisper_transcribe.py.
    """
    # sys.executable = the same Python interpreter (and virtualenv) that runs this app
    result = subprocess.run(
        [sys.executable, WHISPER_WORKER, audio_path, model_name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # The last stderr line is normally the actual error message
        error_lines = result.stderr.strip().splitlines()
        raise RuntimeError(error_lines[-1] if error_lines else f"worker exited with code {result.returncode}")
    # The worker prints its result as JSON on the last line of stdout
    return json.loads(result.stdout.strip().splitlines()[-1])["text"]

def recognize_speech_and_save():
    """Record one utterance from the microphone and transcribe it offline with Whisper.

    Returns the transcribed text, or "" if nothing usable was recorded. The transcript is
    also offered as a .txt download.
    """
    r = sr.Recognizer()
    mic_names = sr.Microphone.list_microphone_names()
    if not mic_names:
        st.error("❌ No microphones detected")
        return ""
    # Prefer a device whose name contains "microphone"; otherwise use the first input device
    mic_index = next((i for i, name in enumerate(mic_names) if "microphone" in name.lower()), 0)
    try:
        with sr.Microphone(device_index=mic_index) as source:
            with st.spinner("🎤 Listening..."):
                # Calibrate the silence threshold, then record until the speaker pauses
                # (gives up if no speech starts within 8 seconds)
                r.adjust_for_ambient_noise(source, duration=0.8)
                audio = r.listen(source, timeout=8)
        with st.spinner("Transcribing speech locally with Whisper..."):
            wav_data = audio.get_wav_data()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                tmp_wav.write(wav_data)
                tmp_wav_path = tmp_wav.name
            try:
                text = run_whisper(tmp_wav_path)
            finally:
                if os.path.exists(tmp_wav_path):
                    os.remove(tmp_wav_path)

        if text:
            st.success(f"Transcribed: {text}")
            st.download_button("Download Text", text, file_name="transcript.txt")
            return text
        else:
            st.warning("❌ No speech detected in audio")
            return ""
    except sr.WaitTimeoutError:
        st.warning("⌛ Listening timed out")
    except Exception as e:
        st.error(f"⚠️ Recognition error: {str(e)}")
    return ""

def transcribe_audio_file(file):
    """Transcribe an uploaded audio file (.mp3/.wav/.m4a) offline with Whisper.

    Returns the text, or "" on failure (the error is shown in the UI).
    """
    if not file:
        return ""

    # Whisper decodes audio with the ffmpeg command-line tool, so check it is installed first
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except Exception:
        st.error("❌ FFmpeg is not installed or not in PATH. Please install FFmpeg.")
        return ""

    # Keep the original extension so ffmpeg can recognise the format
    suffix = f".{file.name.split('.')[-1]}" if hasattr(file, "name") and "." in file.name else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        st.info("🔁 Transcribing offline with Whisper...")
        text = run_whisper(tmp_path)
        if not text:
            st.error("❌ Transcription empty or failed.")
            return ""
        return text
    except Exception as e:
        st.error(f"❌ Whisper Transcription failed: {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# === Chat History Management ===
# Each conversation is stored as chat_sessions/<uuid>.json:
#   {"title": ..., "timestamp": <last update, ISO 8601>, "messages": [{"type": "human"|"ai", "content": ...}]}

def get_all_sessions():
    """Return all saved conversations, most recently updated first.

    Files that cannot be read or parsed are skipped so one bad file cannot break the sidebar.
    """
    sessions = []
    if os.path.exists(HISTORY_DIR):
        for filename in os.listdir(HISTORY_DIR):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(HISTORY_DIR, filename)
                    with open(filepath, 'r') as f:
                        session_data = json.load(f)
                        sessions.append({
                            'id': filename[:-5],
                            'title': session_data.get('title', 'Untitled Chat'),
                            'timestamp': session_data.get('timestamp', ''),
                            'messages': session_data.get('messages', [])
                        })
                except Exception:
                    continue
    # ISO 8601 timestamps sort chronologically as plain strings
    return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)

def save_session(session_id, title, messages):
    """Write the whole conversation to its JSON file (overwriting the previous version)."""
    filepath = os.path.join(HISTORY_DIR, f"{session_id}.json")
    session_data = {
        'title': title,
        'timestamp': datetime.now().isoformat(),
        'messages': [{"type": msg.type if hasattr(msg, 'type') else msg.get('type', 'user'),
                     "content": msg.content if hasattr(msg, 'content') else msg.get('content', '')}
                    for msg in messages]
    }
    try:
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving session: {e}")

def load_session(session_id):
    """Return the saved session dict, or None if the file is missing or unreadable."""
    filepath = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def delete_session(session_id):
    """Permanently delete a saved conversation."""
    filepath = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)

def generate_title_from_message(message):
    """Use the first four words of the first message as the chat title."""
    words = message.split()[:4]
    return " ".join(words) + ("..." if len(message.split()) > 4 else "")

def _supports_chat(ollama_client, model):
    """Exclude embedding-only models (e.g. nomic-embed-text) from the chat model list"""
    try:
        capabilities = ollama_client.show(model).capabilities
    except Exception:
        return True  # capability info unavailable (older Ollama) - keep the model
    return not capabilities or "completion" in capabilities

def get_local_ollama_models():
    """Discover chat-capable models from the local Ollama server (localhost only)"""
    try:
        import ollama
        response = ollama.list()
        # Newer ollama clients return an object, older ones a plain dict
        if hasattr(response, 'models'):
            models = [m.model for m in response.models]
        elif isinstance(response, dict):
            models = [m['name'] for m in response.get('models', [])]
        else:
            models = []
        models = [m for m in models if _supports_chat(ollama, m)]
        if models:
            return models, True
    except Exception:
        pass
    # Ollama is not reachable: offer a default list so the UI still renders, and report "not connected"
    return ["llama3.2:latest", "llama3.2:3b", "mistral:7b"], False

# === Streamlit UI Setup ===
st.set_page_config(page_title="DRDO AI Assistant", page_icon="🤖", layout="wide")

# === Inline CSS (No external file needed) ===
st.markdown("""
<style>
:root {
    --primary: #2e7d32;
    --secondary: #d32f2f;
}
.chat-session-button {
    width: 100%;
    text-align: left;
    padding: 10px;
    margin: 5px 0;
    border-radius: 8px;
    border: 1px solid #ddd;
    background: white;
    cursor: pointer;
}
.chat-session-button:hover {
    background: #f0f0f0;
    border-color: var(--primary);
}
.active-session {
    background: #e8f5e9 !important;
    border-color: var(--primary) !important;
}
.stChatMessage {
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# === Custom Header ===
st.markdown("""
<div style="background: linear-gradient(90deg, #1b5e20 0%, #4682B4 100%);
            padding: 20px;
            border-radius: 0 0 20px 20px;
            color: white;
            text-align: center;">
    <h1>🛡️ DRDO SecureAI Assistant</h1>
    <p>Advanced Document Analysis & Chat System</p>
</div>
""", unsafe_allow_html=True)

# === Initialize Session State ===
# Runs only on the first load of a browser tab; later reruns keep the existing values.
# chat_history (LangChain) is what is sent to the model; messages is what is displayed and saved.
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.messages = []
    st.session_state.session_title = "New Chat"
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = rag.DEFAULT_SYSTEM_PROMPT
if "last_audio_file" not in st.session_state:
    st.session_state.last_audio_file = None

# === Sidebar for Chat History and Controls ===
with st.sidebar:
    st.markdown("### 💬 Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.chat_history = InMemoryChatMessageHistory()
        st.session_state.messages = []
        st.session_state.session_title = "New Chat"
        st.rerun()
    st.markdown("---")
    sessions = get_all_sessions()
    if sessions:
        for session in sessions:
            button_key = f"session_{session['id']}"
            try:
                timestamp = datetime.fromisoformat(session['timestamp'])
                time_str = timestamp.strftime("%b %d, %H:%M")
            except Exception:
                time_str = "Unknown"
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(
                    f"💬 {session['title']}",
                    key=button_key,
                    use_container_width=True,
                    help=f"Last updated: {time_str}"
                ):
                    # Rebuild both in-memory histories from the saved file
                    loaded_session = load_session(session['id'])
                    if loaded_session:
                        st.session_state.current_session_id = session['id']
                        st.session_state.session_title = loaded_session['title']
                        st.session_state.chat_history = InMemoryChatMessageHistory()
                        st.session_state.messages = []
                        for msg in loaded_session['messages']:
                            if msg['type'] == 'human':
                                st.session_state.chat_history.add_user_message(msg['content'])
                                st.session_state.messages.append(HumanMessage(content=msg['content']))
                            else:
                                st.session_state.chat_history.add_ai_message(msg['content'])
                                st.session_state.messages.append(AIMessage(content=msg['content']))
                        st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{session['id']}", help="Delete chat"):
                    delete_session(session['id'])
                    if st.session_state.current_session_id == session['id']:
                        st.session_state.current_session_id = str(uuid.uuid4())
                        st.session_state.chat_history = InMemoryChatMessageHistory()
                        st.session_state.messages = []
                        st.session_state.session_title = "New Chat"
                    st.rerun()
            st.markdown(f"<small style='color: #666;'>{time_str}</small>", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.markdown("*No chat history yet*")
    st.markdown("### ⚙️ Configuration")
    # Model list comes from the local Ollama server; embedding-only models are filtered out
    local_models, ollama_connected = get_local_ollama_models()
    if ollama_connected:
        st.caption("🟢 Local Ollama: Connected")
    else:
        st.caption("🟠 Local Ollama: Run `ollama serve` offline")
    model_name = st.selectbox("Model", local_models, index=0)
    st.session_state.system_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt)
    st.markdown("### 🔊 Voice Features")
    use_voice_input = st.toggle("🎙 Voice Input", value=False)
    use_voice_output = st.toggle("🔊 Voice Output", value=False)
    st.markdown("""
    <div style="background-color: #4682B4;
                padding: 15px;
                border-radius: 12px;
                border-left: 4px solid #2e7d32;
                margin-top: 20px;">
        <h4>🔐 Security Status</h4>
        <p>• Local Processing Only</p>
        <p>• No Data Transmission</p>
        <p>• Chats saved as local JSON (unencrypted)</p>
    </div>
    """, unsafe_allow_html=True)

# === Main Chat Interface ===
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"💬 {st.session_state.session_title}")
with col2:
    uploaded_file = st.file_uploader("📄 Upload", type=["pdf", "txt"])
    audio_file = st.file_uploader("🎙 Upload Audio", type=["mp3", "wav", "m4a"])


# === Document Processing (Cached Locally) ===
# Indexing is expensive, so it only happens when a new file appears; the FAISS index is kept in
# session state and reused on every later rerun. Removing the file drops the index.
doc_store = None
if uploaded_file:
    # Name + size identifies the upload cheaply (a different file with the same name and size is not detected)
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if "current_doc_id" not in st.session_state or st.session_state.current_doc_id != file_id:
        st.session_state.pop("doc_store", None)
        st.session_state.pop("doc_chunks_count", None)
        with st.spinner("📄 Indexing document offline with FAISS..."):
            # The LangChain loaders read from a file path, so write the upload to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getbuffer())
                file_path = tmp.name
            try:
                loader = PyPDFLoader(file_path) if uploaded_file.type == "application/pdf" else TextLoader(file_path, encoding="utf-8")
                docs = loader.load()  # PDFs: one Document per page; TXT: one Document
                # Split into chunks, embed them with nomic-embed-text and build the FAISS index
                st.session_state.doc_store, st.session_state.doc_chunks_count = rag.build_index(docs)
            except Exception as e:
                st.error(f"⚠️ Could not index document: {e}. "
                         f"Make sure the embedding model is installed: `ollama pull {rag.EMBED_MODEL}`")
            finally:
                # Mark this file as processed either way so a failure is not retried on every rerun
                st.session_state.current_doc_id = file_id
                if os.path.exists(file_path):
                    os.remove(file_path)
    doc_store = st.session_state.get("doc_store")
    if doc_store:
        st.success(f"✅ Loaded {st.session_state.get('doc_chunks_count', 0)} document chunks")
elif "current_doc_id" in st.session_state:
    st.session_state.pop("current_doc_id", None)
    st.session_state.pop("doc_store", None)
    st.session_state.pop("doc_chunks_count", None)

# === Display Chat Messages ===
# Redraw the whole conversation on every rerun (Streamlit does not keep earlier output)
for msg in st.session_state.messages:
    if hasattr(msg, 'type'):
        if msg.type == "human":
            st.chat_message("user", avatar="👤").markdown(msg.content)
        else:
            st.chat_message("assistant", avatar="🤖").markdown(msg.content)
    else:
        if isinstance(msg, dict):
            if msg.get("type") == "human":
                st.chat_message("user", avatar="👤").markdown(msg.get("content", ""))
            else:
                st.chat_message("assistant", avatar="🤖").markdown(msg.get("content", ""))

# === Input Handling ===
# Transcribe an uploaded audio file once. Comparing with the last processed upload stops the
# same file from being transcribed again on every rerun (UploadedFile equality uses its file id).
transcribed_input = None
if audio_file and ("last_audio_file" not in st.session_state or st.session_state.last_audio_file != audio_file):
    st.session_state.last_audio_file = audio_file
    st.info("📥 Transcribing audio file using Whisper...")
    transcribed_input = transcribe_audio_file(audio_file)
    if transcribed_input:
        st.success(f"✅ Transcribed Text: {transcribed_input}")
elif not audio_file:
    st.session_state.last_audio_file = None
user_input = None  # initialize early

# Set priority: 1. Audio Transcription, 2. Voice Input, 3. Manual Chat Input
# st.chat_input is called at the top level (not inside a container) so Streamlit pins it to the bottom
if transcribed_input:
    user_input = transcribed_input
elif use_voice_input:
    if st.button("🎤 Press & Speak", use_container_width=True):  # True only on the run right after the click
        user_input = recognize_speech_and_save()
else:
    user_input = st.chat_input("🔒 Enter your secure query...")

# === Chat Processing with Centered Spinner ===
# 1. record the question  2. retrieve document context  3. call the LLM
# 4. show + store the answer  5. optional voice output  6. save the conversation to disk
if user_input:
    st.chat_message("user", avatar="👤").markdown(user_input)
    st.session_state.chat_history.add_user_message(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    if st.session_state.session_title == "New Chat" and len(st.session_state.messages) == 1:
        st.session_state.session_title = generate_title_from_message(user_input)
    llm = ChatOllama(model=model_name)  # talks to Ollama at localhost:11434 (POST /api/chat)
    st.markdown(spinner_css, unsafe_allow_html=True)
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown("<div class='custom-loader'></div>", unsafe_allow_html=True)
    try:
        context_docs = []
        if doc_store:
            # Embed the question, find the closest chunks in FAISS and keep the relevant ones.
            # A retrieval failure is not fatal: the question is answered without document context.
            try:
                context_docs = rag.select_context(rag.retrieve(doc_store, user_input))
            except Exception as e:
                st.error(f"⚠️ Retrieval error: {str(e)}")
        # chat_history already ends with the current question, which build_messages re-adds with context
        messages = rag.build_messages(st.session_state.system_prompt,
                                      st.session_state.chat_history.messages[:-1],
                                      user_input, context_docs)
        response = llm.invoke(messages)
        spinner_placeholder.empty()  # Remove spinner
        st.chat_message("assistant", avatar="🤖").markdown(response.content)
        st.session_state.chat_history.add_ai_message(response.content)
        st.session_state.messages.append(AIMessage(content=response.content))
        if use_voice_output:
            text_to_speech_and_save(response.content)  # audio player + .wav download (waits)
            speak(response.content)                    # read aloud in the background (does not wait)
        save_session(
            st.session_state.current_session_id,
            st.session_state.session_title,
            st.session_state.messages
        )
    except Exception as e:
        # Typically Ollama is not running or the model is missing
        spinner_placeholder.empty()
        st.error(f"Error: {e}")
        fallback = "I apologize, but I'm experiencing technical difficulties."
        # Shown to the user only: storing it would feed a fake reply back to the model as history
        st.chat_message("assistant", avatar="🤖").markdown(fallback)
        # Still save, so the user's question is not lost
        save_session(
            st.session_state.current_session_id,
            st.session_state.session_title,
            st.session_state.messages
        )
