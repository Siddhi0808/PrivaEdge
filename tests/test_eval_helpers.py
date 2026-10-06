"""Unit tests for the evaluation helpers in eval/run_eval.py and the context filter in rag.py.

These run without Ollama: they only exercise pure functions.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

import rag  # noqa: E402
import run_eval  # noqa: E402


def in_doc(distance):
    return {"type": "in_doc", "evidence_distance": distance, "retrieved": [{"distance": distance}]}


def off_topic(closest):
    return {"type": "off_topic", "retrieved": [{"distance": closest}, {"distance": closest + 0.1}]}


class TuneCutoffTests(unittest.TestCase):
    def test_cutoff_is_midpoint_of_the_gap(self):
        cutoff, info = run_eval.tune_cutoff([in_doc(0.5), in_doc(0.7), off_topic(0.9), off_topic(1.1)])
        self.assertEqual(info["objective"], 1.0)
        self.assertAlmostEqual(cutoff, 0.8)

    def test_distances_closer_than_four_decimals_are_not_rounded_together(self):
        # Regression test: rounding candidates to 4 decimals once placed the cutoff above an
        # off-topic question's closest chunk, so that question still received document context.
        rows = [in_doc(0.89980), off_topic(0.927904)]
        cutoff, _ = run_eval.tune_cutoff(rows)
        self.assertGreaterEqual(cutoff, 0.89980)          # evidence kept
        self.assertLess(cutoff, 0.927904)                 # off-topic question gets no context


class GraderTests(unittest.TestCase):
    def test_whole_word_and_number_normalisation(self):
        self.assertTrue(run_eval.is_correct("The chemical symbol is Au.", [["au"]]))
        self.assertFalse(run_eval.is_correct("because of the weather", [["au"]]))
        self.assertTrue(run_eval.is_correct("It replaced 1,240 lines", [["1240"]]))
        self.assertFalse(run_eval.is_correct("In 2024 they did it", [["24"]]))
        self.assertTrue(run_eval.is_correct("Tuesdays and Thursdays", [["tuesday", "tuesdays"], ["thursdays"]]))


class SelectContextTests(unittest.TestCase):
    def test_keeps_only_chunks_within_max_distance(self):
        results = [("a", 0.5), ("b", 0.95), ("c", 1.2)]
        self.assertEqual(rag.select_context(results, None), ["a", "b", "c"])
        self.assertEqual(rag.select_context(results, 0.95), ["a", "b"])
        self.assertEqual(rag.select_context(results, 0.1), [])


if __name__ == "__main__":
    unittest.main()
