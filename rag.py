"""Document Q&A (RAG) pipeline shared by UIChat.py and the evaluation script (eval/run_eval.py).

Split documents into chunks -> embed them locally with Ollama -> index them in FAISS ->
retrieve the chunks closest to a question -> build the message list sent to the chat model.
"""
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chat models such as llama3.2 cannot produce embeddings in current Ollama
# versions, so document indexing uses a dedicated local embedding model.
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000      # characters
CHUNK_OVERLAP = 100    # characters
TOP_K = 3
DEFAULT_SYSTEM_PROMPT = "You are a helpful, witty, and concise assistant."


def build_index(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split LangChain Documents into chunks and index them in an in-memory FAISS store.

    Returns (vector_store, number_of_chunks). Raises ValueError if there is no text to index.
    """
    # Splits on paragraphs first, then lines, then words, so chunks break at natural boundaries;
    # the overlap repeats the end of each chunk at the start of the next one
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("no extractable text found in the document")
    # Embeds every chunk via Ollama and stores the vectors in an exact (brute-force) FAISS index
    return FAISS.from_documents(chunks, OllamaEmbeddings(model=EMBED_MODEL)), len(chunks)


def retrieve(vector_store, question, k=TOP_K):
    """Return the k chunks closest to the question as a list of (Document, distance) pairs.

    The distance is FAISS's squared L2 distance. nomic-embed-text vectors have unit length,
    so distance = 2 - 2 * cosine_similarity (0 = identical, larger = less related).
    """
    return vector_store.similarity_search_with_score(question, k=k)


def select_context(results, max_distance=None):
    """Keep the retrieved chunks whose distance is at most max_distance (None keeps all of them)."""
    return [doc for doc, distance in results if max_distance is None or distance <= max_distance]


def build_messages(system_prompt, history, question, context_docs):
    """Build the message list for the chat model.

    history: previous turns (LangChain messages), excluding the current question.
    context_docs: retrieved Documents whose text is placed in front of the question.
    Retrieved context goes only into the current turn; history keeps the raw questions.
    """
    context = ""
    if context_docs:
        context = "\n\n".join(doc.page_content for doc in context_docs)
        context = f"Document Context:\n{context}\n\n"
    return [SystemMessage(content=system_prompt)] + list(history) + [
        HumanMessage(content=f"{context}User: {question}")
    ]
