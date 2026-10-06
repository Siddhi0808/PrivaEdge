"""Offline AI Assistant - minimal terminal chat.

A text-only chat with the local llama3.2 model that remembers the conversation while the
program runs (nothing is saved to disk). Type 'exit' or 'quit', or press Ctrl-D / Ctrl-C, to leave.

Run with:  python TerminalChat.py   (requires `ollama serve` and `ollama pull llama3.2`)
"""
from langchain_ollama import ChatOllama

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Chat model served by the local Ollama server (http://localhost:11434)
llm = ChatOllama(model="llama3.2")

system_message = SystemMessage(
    content="You are a helpful, witty, and concise personal assistant who speaks in a friendly tone."
)

# Prompt layout for every turn: system instructions, then the conversation so far
# (filled into the "history" placeholder), then the new user message.
prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# In-memory conversation histories, keyed by session id
session_histories = {}

def get_session_history(session_id: str):
    """Return the history for a session, creating an empty one the first time."""
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]

# `prompt | llm` chains the template into the model (LangChain Expression Language).
# RunnableWithMessageHistory wraps that chain: before each call it loads the session's history
# into the "history" placeholder, and afterwards it appends the new question and answer to it.
chatbot = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

print("🤖 Your Personal Chatbot (type 'exit' to quit)\n")

# Read-eval-print loop
while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C: exit cleanly instead of a traceback
        print()
        break
    if not user_input:  # ignore empty lines
        continue
    if user_input.lower() in ["exit", "quit"]:
        break

    try:
        # A single fixed session id: one conversation per program run
        response = chatbot.invoke({"input": user_input}, config={"configurable": {"session_id": "chat1"}})
    except Exception as e:  # e.g. Ollama not running or model not downloaded - keep the loop alive
        print(f"Error: {e}\n(Is Ollama running? Try `ollama serve` and `ollama pull llama3.2`)\n")
        continue
    print("Bot:", response.content, "\n")
