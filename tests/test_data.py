"""Check that released inference inputs cannot accidentally include gold targets."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return json.loads((ROOT / rel).read_text())


def lines(rel):
    return [json.loads(row) for row in (ROOT / rel).read_text().splitlines() if row]


class DataBoundaryTests(unittest.TestCase):
    def test_input_allowlists_and_counts(self):
        for part, n in [('recognition', 5462), ('execution', 666)]:
            rows = lines(f'data/{part}/inputs.jsonl')
            self.assertEqual(len(rows), n)
            self.assertEqual(len({r['id'] for r in rows}), n)
            self.assertTrue(all(set(r) == {'id', 'stream'} for r in rows))
            self.assertTrue(all(isinstance(r['stream'], str) and r['stream'].strip() for r in rows))
            self.assertEqual({r['id'] for r in rows}, {r['id'] for r in lines(f'data/{part}/annotations.jsonl')})

    def test_recognition_fit_banks_never_use_evaluation(self):
        ann = {r['id']: r for r in lines('data/recognition/annotations.jsonl')}
        self.assertEqual(Counter(r['split'] for r in ann.values()), {'fit': 3072, 'eval': 2390})
        banks = read('data/recognition/fit_banks.json')
        self.assertEqual(len(banks), 82)
        for rid, bank in banks.items():
            self.assertEqual(len(bank['positive']), 16)
            self.assertEqual(len(bank['negative']), 1756)
            self.assertFalse(set(bank['positive']) & set(bank['negative']))
            for sid in bank['positive'] + bank['negative']:
                self.assertEqual(ann[sid]['split'], 'fit')
            self.assertTrue(all(ann[sid]['rule_id'] == rid and ann[sid]['kind'] == 'applicable' for sid in bank['positive']))
        text = {r['id']: ' '.join(r['stream'].split()) for r in lines('data/recognition/inputs.jsonl')}
        fit = {text[sid] for sid, r in ann.items() if r['split'] == 'fit'}
        evaluation = {text[sid] for sid, r in ann.items() if r['split'] == 'eval'}
        self.assertFalse(fit & evaluation)

    def test_execution_split_and_reference_boundaries(self):
        train = lines('data/execution/train.jsonl')
        val = lines('data/execution/validation.jsonl')
        ann = lines('data/execution/annotations.jsonl')
        self.assertEqual((len(train), len(val)), (491, 164))
        self.assertEqual(Counter(r['split'] for r in ann), {'development': 112, 'main': 464, 'seed': 90})
        ids = [{r['id'] for r in rows} for rows in [train, val, ann]]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(len(read('data/rules.json')), 82)
        self.assertEqual(len(lines('data/execution/value_train.jsonl')), 80)

    def test_original_known_issues_preserved(self):
        text = {r['id']: r['stream'] for r in lines('data/recognition/inputs.jsonl')}
        ann = {r['id']: r for r in lines('data/recognition/annotations.jsonl')}
        a, b = 'transformed:r051-p03', 'transformed:r051-h02'
        self.assertEqual(text[a], text[b])
        self.assertNotEqual(ann[a]['kind'], ann[b]['kind'])
        self.assertEqual(read('data/recognition/audit.json')['normalized_fit_eval_overlap'], 0)

    def test_protocol_is_known_collection_not_unknown_arrivals(self):
        protocol = read('data/protocols/execution_installation.json')
        self.assertTrue(protocol['recognizer_uses_full_known_rule_collection'])
        self.assertFalse(protocol['future_value_states_available'])
        self.assertFalse(protocol['prior_responses_enter_next_input'])
        self.assertEqual([r['installed'] for r in protocol['stages']], list(range(83)))
        for stage in protocol['stages']:
            self.assertEqual(stage['rule_ids'], protocol['rule_order'][:stage['installed']])
        replay = read('data/protocols/recognition_replay.json')
        self.assertEqual(len(replay['snapshots']), 492)
        self.assertTrue(replay['full_contrastive_bank_known_before_replay'])

    def test_frozen_bundle_hashes_and_case_refs_when_downloaded(self):
        manifest = ROOT / 'artifacts/frozen/manifest.json'
        if not manifest.exists():
            self.skipTest('optional frozen evidence asset has not been downloaded')
        for rel, meta in read('artifacts/frozen/manifest.json')['files'].items():
            raw = (ROOT / rel).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), meta['sha256'], rel)
        outputs = {r['output_id']: r for r in lines('artifacts/frozen/execution/outputs.jsonl')}
        packets = {r['judge_packet_id']: r for r in lines('artifacts/frozen/execution/judge_packets.jsonl')}
        for row in outputs.values():
            self.assertIn(row['judge_packet_id'], packets)
            for label in ('action', 'positive', 'repetition'):
                self.assertEqual(row['labels'][label], packets[row['judge_packet_id']]['labels'][label])
        for case in lines('artifacts/frozen/execution/cases.jsonl'):
            self.assertEqual(outputs[case['output_id']]['sample_id'], case['sample_id'])
        stages = lines('artifacts/frozen/execution/stages.jsonl')
        self.assertEqual(Counter(r['method'] for r in stages), {'siren_kv': 83, 'pi_v1': 83, 'pi_v2': 83})
        for stage in stages:
            self.assertEqual(len(stage['system']), 666)
            self.assertTrue(set(stage['system'].values()).issubset(outputs))


if __name__ == '__main__':
    unittest.main()
