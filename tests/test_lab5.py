"""Unit tests (no network needed): python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab5 import (
    get_count, get_probs, delete_letter, switch_letter, replace_letter,
    insert_letter, edit_one_letter, edit_two_letters, get_corrections,
    min_edit_distance, correct_text, generate_test_cases, evaluate
)


class TestLab5(unittest.TestCase):
    def test_frequency_and_probabilities(self):
        counts = get_count('hello hello world'.split())
        self.assertEqual(counts, {'hello': 2, 'world': 1})
        self.assertAlmostEqual(sum(get_probs(counts).values()), 1.0)
        self.assertEqual(get_probs({}), {})

    def test_delete(self):
        self.assertEqual(set(delete_letter('hat')), {'at', 'ht', 'ha'})

    def test_switch(self):
        self.assertIn('tea', switch_letter('eta'))

    def test_replace(self):
        self.assertIn('hat', replace_letter('jat'))
        self.assertNotIn('jat', replace_letter('jat'))

    def test_insert(self):
        self.assertIn('the', insert_letter('te'))

    def test_edits_one_two(self):
        self.assertIn('hello', edit_one_letter('hllo'))
        self.assertIn('hello', edit_two_letters('hlo', vocab={'hello', 'world'}))

    def test_correction_ranking(self):
        counts = {'hello': 20, 'hallo': 2, 'world': 10, 'money': 20, 'honey': 5}
        vocab = set(counts)
        probs = get_probs(counts)
        self.assertEqual(get_corrections('hllo', probs, vocab)[0][0], 'hello')
        self.assertEqual(get_corrections('hello', probs, vocab)[0][0], 'hello')
        self.assertEqual(correct_text('Hllo, WORLD!', probs, vocab), 'Hello, WORLD!')

    def test_min_edit_distance(self):
        D, d = min_edit_distance('play', 'stay')
        self.assertEqual(d, 4)  # p→s and l→t, each costs 2
        self.assertEqual(D.shape, (5, 5))
        self.assertEqual(min_edit_distance('', 'cat')[1], 3)
        self.assertEqual(min_edit_distance('sunday', 'saturday')[1], 4)
        self.assertEqual(min_edit_distance('abc', 'abc')[1], 0)

    def test_synthetic_evaluation(self):
        counts = get_count(('hello '*10 + 'world '*10 + 'every '*10 + 'other '*10 +
                           'words '*10 + 'house '*10 + 'there '*10 + 'friend '*10).split())
        cases = generate_test_cases(counts, 4, seed=8)
        self.assertEqual(len(cases), 4)
        self.assertTrue(all(c['typed'] not in counts for c in cases))
        metrics, predictions = evaluate(cases, get_probs(counts), set(counts))
        self.assertEqual(metrics['test_count'], len(predictions))
        self.assertGreaterEqual(metrics['accuracy_top3'], metrics['accuracy_top1'])


if __name__ == '__main__':
    unittest.main()
