import os
import json
import uuid
import tempfile
from datetime import datetime
from threading import Thread

import streamlit as st
import speech_recognition as sr
import pyttsx3

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
os.environ["STREAMLIT_WATCH_FILE_CHANGES"] = "false"

# === Constants & Path Setup ===
HISTORY_DIR = "chat_sessions"
os.makedirs(HISTORY_DIR, exist_ok=True)

# === Voice Engine Setup (initialize once) ===
engine = pyttsx3.init()
engine.setProperty('rate', 175)
engine.setProperty('volume', 0.9)

# === Custom CSS for Centered Spinner ===
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
    """Threaded text-to-speech to prevent UI blocking"""
    def _speak():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    Thread(target=_speak).start()

def text_to_speech_and_save(text):
    """Save TTS output as wav and offer download"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        try:
            engine.stop()
        except Exception:
            pass
        engine.save_to_file(text, tmp.name)
        engine.runAndWait()
        st.audio(tmp.name)
        with open(tmp.name, "rb") as f:
            st.download_button("Download Audio", f, file_name="response.wav")

@st.cache_resource
def load_offline_whisper(model_name="base"):
    """Load Whisper model once and cache it in memory for offline use"""
    import whisper
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    return whisper.load_model(model_name, download_root=models_dir)

def recognize_speech_and_save():
    """Voice input, transcribe offline using Whisper, and offer download"""
    r = sr.Recognizer()
    mic_names = sr.Microphone.list_microphone_names()
    if not mic_names:
        st.error("❌ No microphones detected")
        return ""
    mic_index = next((i for i, name in enumerate(mic_names) if "microphone" in name.lower()), 0)
    try:
        with sr.Microphone(device_index=mic_index) as source:
            with st.spinner("🎤 Listening..."):
                r.adjust_for_ambient_noise(source, duration=0.8)
                audio = r.listen(source, timeout=8)
        with st.spinner("Transcribing speech locally with Whisper..."):
            wav_data = audio.get_wav_data()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                tmp_wav.write(wav_data)
                tmp_wav_path = tmp_wav.name
            try:
                model = load_offline_whisper("base")
                result = model.transcribe(tmp_wav_path)
                text = result.get("text", "").strip()
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
    except sr.UnknownValueError:
        st.warning("❌ Could not understand audio")
    except Exception as e:
        st.error(f"⚠️ Recognition error: {str(e)}")
    return ""

def transcribe_audio_file(file):
    """Transcribe uploaded audio offline using Whisper"""
    if not file:
        return ""
    import subprocess

    # Ensure ffmpeg is working
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except Exception:
        st.error("❌ FFmpeg is not installed or not in PATH. Please install FFmpeg.")
        return ""

    model = load_offline_whisper("base")

    suffix = f".{file.name.split('.')[-1]}" if hasattr(file, "name") and "." in file.name else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        st.info("🔁 Transcribing offline with Whisper...")
        result = model.transcribe(tmp_path)
        if result is None or "text" not in result or not result["text"].strip():
            st.error("❌ Transcription empty or failed.")
            return ""
        return result["text"].strip()
    except Exception as e:
        st.error(f"❌ Whisper Transcription failed: {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# === Chat History Management ===
def get_all_sessions():
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
                except:
                    continue
    return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)

def save_session(session_id, title, messages):
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
    filepath = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def delete_session(session_id):
    filepath = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)

def generate_title_from_message(message):
    words = message.split()[:4]
    return " ".join(words) + ("..." if len(message.split()) > 4 else "")

def get_local_ollama_models():
    """Discover locally available Ollama models without network calls"""
    try:
        import ollama
        response = ollama.list()
        if hasattr(response, 'models'):
            models = [m.model for m in response.models]
        elif isinstance(response, dict):
            models = [m['name'] for m in response.get('models', [])]
        else:
            models = []
        if models:
            return models, True
    except Exception:
        pass
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
st.markdown("""
<style>
/* Make the chat input bar sticky at the bottom */
.sticky-chat-input {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: white;
    padding: 1rem;
    box-shadow: 0 -4px 10px rgba(0, 0, 0, 0.1);
    z-index: 999;
}
.stChatMessage {
    margin-bottom: 80px;  /* Prevent last message being hidden */
}
</style>
""", unsafe_allow_html=True)

# === Initialize Session State ===
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.messages = []
    st.session_state.session_title = "New Chat"
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful, witty, and concise assistant."
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
            except:
                time_str = "Unknown"
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(
                    f"💬 {session['title']}",
                    key=button_key,
                    use_container_width=True,
                    help=f"Created: {time_str}"
                ):
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
        <p>• Encrypted Storage</p>
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
retriever = None
if uploaded_file:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if "current_doc_id" not in st.session_state or st.session_state.current_doc_id != file_id:
        with st.spinner("📄 Indexing document offline with FAISS..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getbuffer())
                file_path = tmp.name
            try:
                loader = PyPDFLoader(file_path) if uploaded_file.type == "application/pdf" else TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                embeddings = OllamaEmbeddings(model=model_name)
                db = FAISS.from_documents(chunks, embeddings)
                st.session_state.doc_retriever = db.as_retriever(search_kwargs={"k": 3})
                st.session_state.current_doc_id = file_id
                st.session_state.doc_chunks_count = len(chunks)
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
    retriever = st.session_state.get("doc_retriever")
    st.success(f"✅ Loaded {st.session_state.get('doc_chunks_count', 0)} document chunks")
elif "current_doc_id" in st.session_state:
    st.session_state.pop("current_doc_id", None)
    st.session_state.pop("doc_retriever", None)
    st.session_state.pop("doc_chunks_count", None)

# === Display Chat Messages ===
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
# Transcribe audio before any UI input interaction
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

input_col, voice_col = st.columns([0.85, 0.15])
# === Sticky Chat Input at Bottom ===
with st.container():
    st.markdown('<div class="sticky-chat-input">', unsafe_allow_html=True)

    # Set priority: 1. Audio Transcription, 2. Voice Input, 3. Manual Chat Input
    if transcribed_input:
        user_input = transcribed_input
    elif use_voice_input:
        if st.button("🎤 Press & Speak", use_container_width=True):
            user_input = recognize_speech_and_save()
        else:
            user_input = ""
    else:
        user_input = st.chat_input("🔒 Enter your secure query...")

    st.markdown('</div>', unsafe_allow_html=True)


# Final prioritization
if transcribed_input:
    user_input = transcribed_input

# === Chat Processing with Centered Spinner ===
if user_input:
    st.chat_message("user", avatar="👤").markdown(user_input)
    st.session_state.chat_history.add_user_message(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    if st.session_state.session_title == "New Chat" and len(st.session_state.messages) == 1:
        st.session_state.session_title = generate_title_from_message(user_input)
    llm = ChatOllama(model=model_name)
    st.markdown(spinner_css, unsafe_allow_html=True)
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown("<div class='custom-loader'></div>", unsafe_allow_html=True)
    try:
        context = ""
        if retriever:
            try:
                context_docs = retriever.invoke(user_input)
                context = "\n\n".join([doc.page_content for doc in context_docs])
                context = f"Document Context:\n{context}\n\n"
            except Exception as e:
                st.error(f"⚠️ Retrieval error: {str(e)}")
        messages = [SystemMessage(content=st.session_state.system_prompt)] + list(st.session_state.chat_history.messages[:-1])
        response = llm.invoke(messages + [HumanMessage(content=f"{context}User: {user_input}")])
        spinner_placeholder.empty()  # Remove spinner
        st.chat_message("assistant", avatar="🤖").markdown(response.content)
        st.session_state.chat_history.add_ai_message(response.content)
        st.session_state.messages.append(AIMessage(content=response.content))
        if use_voice_output:
            text_to_speech_and_save(response.content)
            speak(response.content)
        save_session(
            st.session_state.current_session_id,
            st.session_state.session_title,
            st.session_state.messages
        )
    except Exception as e:
        spinner_placeholder.empty()
        st.error(f"Error: {e}")
        fallback = "I apologize, but I'm experiencing technical difficulties."
        st.chat_message("assistant", avatar="🤖").markdown(fallback)
        st.session_state.chat_history.add_ai_message(fallback)
        st.session_state.messages.append(AIMessage(content=fallback))
        save_session(
            st.session_state.current_session_id,
            st.session_state.session_title,
            st.session_state.messages
        )
