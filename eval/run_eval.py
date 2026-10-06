"""Evaluate the document Q&A (RAG) pipeline in rag.py on the labelled set in eval/questions.json.

Usage (from the project root, with Ollama running):
    python eval/run_eval.py

What it measures
- Retrieval (in-document questions): Hit@1, Hit@3 and MRR@3, where a hit means a retrieved
  chunk contains the labelled evidence sentence.
- Answer accuracy: the chat model's reply contains the expected answer terms
  (see "answer" in questions.json). Measured separately for in-document questions and for
  off-topic general-knowledge questions asked while a document is loaded.
- Context injection: how often document chunks are added to the prompt.

Procedure
1. Retrieve the top-k chunks (with distances) for every question.
2. Choose a relevance cutoff using the DEV split only (retrieval statistics, no LLM).
3. Generate answers with and without the cutoff on both splits and grade them.
Results are written to eval/results/.

The chat model is run with temperature 0 and a fixed seed so runs are repeatable; the app
itself uses Ollama's default sampling settings.
"""
import json
import os
import re
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langchain_community.document_loaders import TextLoader  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

import rag  # noqa: E402

EVAL_DIR = os.path.join(ROOT, "eval")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
CHAT_MODEL = "llama3.2:latest"
# A reply "mentions the document" if it refers to the loaded document or its contents, e.g.
# "we've taken a detour from the Kestrel drone program!" in answer to an unrelated question.
DOC_MENTION = re.compile(r"kestrel|northwind|riverton|handbook|document|context|provided text|text you provided",
                         re.IGNORECASE)


def normalize(text):
    text = text.lower().replace("µ", "u").replace("μ", "u")
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 1,240 -> 1240


def contains_term(text, term):
    # Whole-word match so that e.g. "au" does not match "because" and "24" does not match "2024"
    return re.search(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])", text) is not None


def is_correct(reply, answer_groups):
    text = normalize(reply)
    return all(any(contains_term(text, term) for term in group) for group in answer_groups)


def load_dataset():
    with open(os.path.join(EVAL_DIR, "questions.json")) as f:
        data = json.load(f)
    docs = {name: TextLoader(os.path.join(EVAL_DIR, path), encoding="utf-8").load()
            for name, path in data["docs"].items()}
    return data["questions"], docs


def run_retrieval(questions, docs):
    """Index each document once and retrieve the top-k chunks for every question."""
    stores, chunk_counts = {}, {}
    for name, doc in docs.items():
        stores[name], chunk_counts[name] = rag.build_index(doc)
    rows = []
    for q in questions:
        results = rag.retrieve(stores[q["doc"]], q["question"])
        row = {**q, "retrieved": [{"text": d.page_content, "distance": float(s)} for d, s in results]}
        if q["type"] == "in_doc":
            ranks = [i + 1 for i, r in enumerate(row["retrieved"]) if q["evidence"] in r["text"]]
            row["evidence_rank"] = ranks[0] if ranks else None
            row["evidence_distance"] = row["retrieved"][ranks[0] - 1]["distance"] if ranks else None
        rows.append(row)
    return rows, chunk_counts


def retrieval_metrics(rows):
    in_doc = [r for r in rows if r["type"] == "in_doc"]
    n = len(in_doc)
    return {
        "n": n,
        "hit_at_1": sum(r["evidence_rank"] == 1 for r in in_doc) / n,
        "hit_at_3": sum(r["evidence_rank"] is not None for r in in_doc) / n,
        "mrr_at_3": sum(1 / r["evidence_rank"] for r in in_doc if r["evidence_rank"]) / n,
    }


def tune_cutoff(dev_rows):
    """Pick the distance cutoff on the dev split.

    Objective: balanced accuracy of two goals, using retrieval only (no LLM):
      - keep the evidence chunk for in-document questions,
      - add no context at all for off-topic questions.
    Among cutoffs that reach the best score, the midpoint of that range is used (largest margin).
    """
    in_doc = [r for r in dev_rows if r["type"] == "in_doc"]
    off = [r for r in dev_rows if r["type"] == "off_topic"]
    # Exact distances (no rounding): a rounded candidate can fall just below a real distance
    # and make a question look correctly separated when it is not
    candidates = sorted({x["distance"] for r in dev_rows for x in r["retrieved"]})
    candidates = [0.0] + candidates + [4.0]

    def score(t):
        kept = sum(r["evidence_distance"] is not None and r["evidence_distance"] <= t for r in in_doc) / len(in_doc)
        clean = sum(min(x["distance"] for x in r["retrieved"]) > t for r in off) / len(off)
        return 0.5 * kept + 0.5 * clean, kept, clean

    scored = [(t, *score(t)) for t in candidates]
    best = max(s[1] for s in scored)
    best_ts = [s[0] for s in scored if s[1] == best]
    # midpoint between the lowest best cutoff and the next candidate above the highest best cutoff
    lo, hi = min(best_ts), max(best_ts)
    above = [t for t in candidates if t > hi]
    cutoff = (lo + (above[0] if above else hi)) / 2
    return cutoff, {"objective": best, "best_range": [lo, above[0] if above else hi], "curve": scored}


def answer_all(rows, llm, max_distance):
    out = []
    for r in rows:
        context_docs = rag.select_context([(Doc(x["text"]), x["distance"]) for x in r["retrieved"]], max_distance)
        messages = rag.build_messages(rag.DEFAULT_SYSTEM_PROMPT, [], r["question"], context_docs)
        start = time.time()
        reply = llm.invoke(messages).content
        out.append({
            "id": r["id"], "split": r["split"], "type": r["type"], "question": r["question"],
            "context_chunks": len(context_docs),
            "evidence_in_context": (r.get("evidence") is not None
                                    and any(r["evidence"] in d.page_content for d in context_docs)),
            "reply": reply, "correct": is_correct(reply, r["answer"]),
            "mentions_document": bool(DOC_MENTION.search(reply)),
            "latency_s": round(time.time() - start, 2),
        })
    return out


class Doc:
    """Minimal stand-in for a LangChain Document (only page_content is used)."""
    def __init__(self, text):
        self.page_content = text


def answer_metrics(answers, split):
    rows = [a for a in answers if a["split"] == split]
    in_doc = [a for a in rows if a["type"] == "in_doc"]
    off = [a for a in rows if a["type"] == "off_topic"]
    return {
        "in_doc_accuracy": sum(a["correct"] for a in in_doc) / len(in_doc),
        "off_topic_accuracy": sum(a["correct"] for a in off) / len(off),
        "overall_accuracy": sum(a["correct"] for a in rows) / len(rows),
        "in_doc_evidence_in_context": sum(a["evidence_in_context"] for a in in_doc) / len(in_doc),
        "off_topic_with_context": sum(a["context_chunks"] > 0 for a in off) / len(off),
        "off_topic_mentions_document": sum(a["mentions_document"] for a in off) / len(off),
        "median_latency_s": statistics.median(a["latency_s"] for a in rows),
        "n_in_doc": len(in_doc), "n_off_topic": len(off),
    }


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    questions, docs = load_dataset()

    print("Retrieving...")
    rows, chunk_counts = run_retrieval(questions, docs)
    retrieval = {split: retrieval_metrics([r for r in rows if r["split"] == split]) for split in ("dev", "test")}
    retrieval["all"] = retrieval_metrics(rows)

    cutoff, tuning = tune_cutoff([r for r in rows if r["split"] == "dev"])
    print(f"Cutoff chosen on dev: distance <= {cutoff}")

    llm = ChatOllama(model=CHAT_MODEL, temperature=0, seed=42)
    llm.invoke("warm up")
    print("Answering without cutoff (baseline)...")
    baseline = answer_all(rows, llm, max_distance=None)
    print("Answering with cutoff...")
    improved = answer_all(rows, llm, max_distance=cutoff)

    summary = {
        "settings": {"chat_model": CHAT_MODEL, "temperature": 0, "seed": 42, "embed_model": rag.EMBED_MODEL,
                     "chunk_size": rag.CHUNK_SIZE, "chunk_overlap": rag.CHUNK_OVERLAP, "top_k": rag.TOP_K,
                     "chunks_per_doc": chunk_counts},
        "retrieval": retrieval,
        "cutoff": cutoff,
        "tuning": {k: v for k, v in tuning.items() if k != "curve"},
        "baseline": {s: answer_metrics(baseline, s) for s in ("dev", "test")},
        "with_cutoff": {s: answer_metrics(improved, s) for s in ("dev", "test")},
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "details.json"), "w") as f:
        json.dump({"retrieval": [{k: v for k, v in r.items() if k != "retrieved"} | {
                       "distances": [round(x["distance"], 4) for x in r["retrieved"]]} for r in rows],
                   "tuning_curve": tuning["curve"], "baseline": baseline, "with_cutoff": improved}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nTEST split (held out):")
    for name in ("baseline", "with_cutoff"):
        m = summary[name]["test"]
        print(f"  {name:12s} in-doc acc {pct(m['in_doc_accuracy'])} | off-topic acc {pct(m['off_topic_accuracy'])} "
              f"| overall {pct(m['overall_accuracy'])} | off-topic with context {pct(m['off_topic_with_context'])} "
              f"| off-topic mentioning the document {pct(m['off_topic_mentions_document'])} "
              f"| evidence kept {pct(m['in_doc_evidence_in_context'])}")


if __name__ == "__main__":
    main()
