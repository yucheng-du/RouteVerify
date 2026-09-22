"""Regression checks for raw-prompt exclusion from publication copies."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from artifact_policy import has_dataset_text, strip_rendered_prompt_text


class ArtifactPolicyTest(unittest.TestCase):
    def test_redaction_preserves_scores_and_identity(self):
        retained = {
            'sample_id': 'synthetic/pair-0/clean',
            'scores': {'CORRECT': {'mean_logprob': -0.5}},
            'input_provenance': {
                'rendered_prompt_sha256': '0' * 64,
                'candidate_contracts': {'CORRECT': {'target_ids': [1, 2]}},
            },
        }
        artifact = copy.deepcopy(retained)
        artifact['input_provenance']['rendered_prompt_text'] = 'Synthetic excluded input.'
        self.assertTrue(has_dataset_text(artifact))
        self.assertEqual(strip_rendered_prompt_text(artifact), 1)
        self.assertEqual(artifact, retained)
        self.assertFalse(has_dataset_text(artifact))

    def test_nested_records(self):
        artifact = [{'nested': {'rendered_prompt_text': 'Synthetic excluded input.'}}]
        self.assertTrue(has_dataset_text(artifact))
        self.assertEqual(strip_rendered_prompt_text(artifact), 1)
        self.assertEqual(artifact, [{'nested': {}}])

    def test_empty_values_and_templates_are_not_artifact_leaks(self):
        self.assertFalse(has_dataset_text({'rendered_prompt_text': None}))
        self.assertFalse(has_dataset_text({'rendered_prompt_text': ''}))
        self.assertFalse(has_dataset_text({'template': 'Choose CORRECT or INCORRECT.'}))


if __name__ == '__main__':
    unittest.main()
