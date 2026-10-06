"""Repeat the answer evaluation with the app's real generation settings (Ollama's default sampling).

run_eval.py uses temperature 0 so results are repeatable. The app does not set a temperature,
so answers vary from run to run. This script measures, with and without the relevance cutoff
chosen by run_eval.py (read from eval/results/summary.json):
  - off-topic questions, each asked REPEATS times: accuracy, and how often the reply talks about
    the loaded document instead of just answering ("distracted" replies),
  - in-document questions, each asked once: accuracy.

Usage (after run_eval.py):  python eval/sampling_check.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import ChatOllama  # noqa: E402

import run_eval  # noqa: E402  (reuses the dataset loading, retrieval and grading code)
import rag  # noqa: E402

REPEATS = 5


def main():
    with open(os.path.join(run_eval.RESULTS_DIR, "summary.json")) as f:
        cutoff = json.load(f)["cutoff"]
    questions, docs = run_eval.load_dataset()
    rows, _ = run_eval.run_retrieval(questions, docs)
    llm = ChatOllama(model=run_eval.CHAT_MODEL)  # default sampling, exactly like UIChat.py
    llm.invoke("warm up")

    records = []
    for config, max_distance in (("baseline", None), ("with_cutoff", cutoff)):
        print(f"Answering ({config})...", flush=True)
        for r in rows:
            repeats = REPEATS if r["type"] == "off_topic" else 1
            pairs = [(run_eval.Doc(x["text"]), x["distance"]) for x in r["retrieved"]]
            context_docs = rag.select_context(pairs, max_distance)
            messages = rag.build_messages(rag.DEFAULT_SYSTEM_PROMPT, [], r["question"], context_docs)
            for i in range(repeats):
                reply = llm.invoke(messages).content
                records.append({"config": config, "id": r["id"], "type": r["type"], "repeat": i,
                                "context_chunks": len(context_docs), "reply": reply,
                                "correct": run_eval.is_correct(reply, r["answer"]),
                                "mentions_document": bool(run_eval.DOC_MENTION.search(reply))})

    summary = {"cutoff": cutoff, "repeats_off_topic": REPEATS}
    for config in ("baseline", "with_cutoff"):
        off = [x for x in records if x["config"] == config and x["type"] == "off_topic"]
        ind = [x for x in records if x["config"] == config and x["type"] == "in_doc"]
        summary[config] = {
            "off_topic_n": len(off),
            "off_topic_accuracy": sum(x["correct"] for x in off) / len(off),
            "off_topic_mentions_document": sum(x["mentions_document"] for x in off) / len(off),
            "in_doc_n": len(ind),
            "in_doc_accuracy": sum(x["correct"] for x in ind) / len(ind),
            "wrong_off_topic_examples": [x["reply"][:200] for x in off if not x["correct"]][:5],
        }
    with open(os.path.join(run_eval.RESULTS_DIR, "sampling_check.json"), "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
