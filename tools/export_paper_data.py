#!/usr/bin/env python3
"""Export an immutable, relocatable paper bundle from an author's local archive.

This is an author-side packaging utility. Readers use the committed data and
downloadable frozen bundle; they do not need the private research repository.
No fitting, generation, judge calls, credentials, or checkpoints are involved.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


METHODS = {
    'l12_frozen': 'siren', 'minilm_lda': 'minilm_discriminant',
    'minilm_maxsim_loo': 'minilm_maxsim', 'minilm_maxsim_self': 'minilm_maxsim_self',
    'numeric': 'siren_kv', 'pi': 'pi_v1', 'pi_improved': 'pi_v2', 'base': 'base',
}
GROUPS = {'silver_development': 'development', 'silver_evaluation': 'main', 'gold_legacy': 'seed'}
KINDS = {'positive': 'applicable', 'hard_negative': 'hard_negative', 'background': 'background'}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Exporter:
    def __init__(self, source_repo: Path, results_root: Path, output_root: Path):
        self.repo = source_repo.resolve()
        self.results = results_root.resolve()
        self.out = output_root.resolve()
        if self.out == self.repo or self.out == self.results or self.out.is_relative_to(self.results):
            raise ValueError('output must be separate from read-only source repository/results')
        self.sources: dict[str, dict] = {}
        self.exports: dict[str, dict] = {}
        self.missing: list[str] = []

    def source_id(self, path: Path) -> str:
        path = path.resolve()
        for root, prefix in ((self.results, 'source-results'), (self.repo, 'source-repo')):
            if path.is_relative_to(root):
                return prefix + '/' + path.relative_to(root).as_posix()
        raise ValueError('source outside declared archive roots')

    def raw(self, path: Path) -> bytes:
        data = path.read_bytes()
        self.sources[self.source_id(path)] = {'sha256': digest(data), 'bytes': len(data)}
        return data

    def read(self, path: Path):
        return json.loads(self.raw(path))

    def clean(self, value):
        """Relocate metadata paths; no stripping or rewriting of example text."""
        if isinstance(value, dict):
            return {k: self.clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if isinstance(value, str):
            for root, prefix in ((str(self.results), 'source-results'), (str(self.repo), 'source-repo')):
                if value.startswith(root + '/'):
                    return prefix + value[len(root):]
            if value.startswith(('/workspace/', '/root/', '/Users/')):
                # Legacy metadata can point at the old machine, not this one.
                for marker, prefix in (('/SIREN-TTCL-results/', 'source-results/'),
                                       ('/SIREN-TTCL-discovery/results/', 'source-results/'),
                                       ('/SIREN-TTCL/', 'source-repo/')):
                    if marker in value:
                        return prefix + value.split(marker, 1)[1]
                return 'source-runtime/' + value.lstrip('/').split('/', 1)[-1]
        return value

    def write_bytes(self, rel: str, data: bytes):
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.exports[rel] = {'sha256': digest(data), 'bytes': len(data)}

    def write(self, rel: str, obj, *, lines=False):
        obj = self.clean(obj)
        if lines:
            data = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n' for row in obj)
        else:
            data = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
        self.write_bytes(rel, data.encode())

    def results_path(self, rel: str) -> Path:
        return self.results / 'siren100' / rel

    def read_run(self, run: str, rel: str):
        return self.read(self.results_path(run) / rel)

    def export_data(self):
        run = 'e30_minilm_system_v1'
        corpus = self.read_run(run, 'corpus.json')
        assert len(corpus) == 5462
        self.write('data/recognition/inputs.jsonl', [
            {'id': r['uid'], 'stream': r['text']} for r in corpus], lines=True)
        annotations = [{'id': r['uid'], 'split': r['split'], 'view': r['view'],
                        'kind': KINDS[r['kind']], 'rule_id': r['rule_id']} for r in corpus]
        assert Counter(r['split'] for r in annotations) == {'fit': 3072, 'eval': 2390}
        self.write('data/recognition/annotations.jsonl', annotations, lines=True)
        self.write('data/recognition/fit_banks.json', self.read_run(run, 'fit_banks.json'))
        self.write('data/recognition/audit.json', self.read_run(run, 'data_audit.json'))
        order = self.read_run('e26_pi_template_control_v1', 'input/order.json')
        train = self.read_run('e24_sequential_r13_v1', 'input/train.json')['rules']
        rules = {r['rule_id']: r for r in train}
        assert len(order) == len(rules) == 82 and 'r033' not in order
        self.write('data/rules.json', [
            {'rule_id': rid, 'condition': rules[rid]['condition'], 'guidance': rules[rid]['thought']}
            for rid in order])
        for part, expected in [('train', 491), ('validation', 164)]:
            rows = [{'rule_id': r['rule_id'], **x} for r in train for x in r[part]]
            assert len(rows) == expected
            self.write(f'data/execution/{part}.jsonl', rows, lines=True)
        legacy = self.read(self.repo / 'data/e5/actions_v0.json')
        value_rows = [{'id': f"{r['rule_id']}-value-fit-{i:02}", 'rule_id': r['rule_id'],
                       'stream': item['stream'], 'action': item['action'], 'thought': r['thought'],
                       'evidence_index': item['evidence_index']}
                      for r in legacy['rules'] for i, item in enumerate(r['items'][:8])]
        assert len(value_rows) == 80
        self.write('data/execution/value_train.jsonl', value_rows, lines=True)
        scenes = self.read_run('e24_sequential_r13_v1', 'input/evaluation.json')['samples']
        assert len(scenes) == 666
        self.write('data/execution/inputs.jsonl', [{'id': s['id'], 'stream': s['stream']} for s in scenes], lines=True)
        ann = [{'id': s['id'], 'rule_id': s['rule_id'], 'split': GROUPS[s['group']],
                'condition': s['condition'], 'required_behavior': s['required_behavior']} for s in scenes]
        assert Counter(s['split'] for s in ann) == {'development': 112, 'main': 464, 'seed': 90}
        self.write('data/execution/annotations.jsonl', ann, lines=True)
        self.write('data/execution/splits.json', {
            split: {'sample_ids': [s['id'] for s in ann if s['split'] == split],
                    'rule_ids': sorted({s['rule_id'] for s in ann if s['split'] == split})}
            for split in ('development', 'main', 'seed')})
        self.write('data/protocols/execution_installation.json', {
            'protocol': 'known_collection_fixed_recognizer_execution_installation',
            'rule_order': order, 'stages': [{'installed': n, 'rule_ids': order[:n]} for n in range(83)],
            'independent_contexts': True, 'prior_responses_enter_next_input': False,
            'future_value_states_available': False,
            'recognizer_uses_full_known_rule_collection': True})
        return order

    def export_routing(self, order):
        run = 'e30_minilm_system_v1'
        for source_method in ('l12_frozen', 'minilm_lda', 'minilm_maxsim_loo', 'minilm_maxsim_self'):
            method = METHODS[source_method]
            rows = self.read_run(run, f'predictions_{source_method}.json')
            rows = [{**r, 'id': r['uid'], 'method': method, 'kind': KINDS[r['kind']]} for r in rows]
            self.write(f'artifacts/frozen/routing/predictions_{method}.jsonl', rows, lines=True)
            source = self.results_path(run) / f'scores_{source_method}.npz'
            self.write_bytes(f'artifacts/frozen/routing/scores_{method}.npz', self.raw(source))
        # The score arrays retain the exact original NPZ keys and row order.
        corpus = self.read_run(run, 'corpus.json')
        self.write('artifacts/frozen/routing/index.json', {
            'sample_ids': [r['uid'] for r in corpus if r['split'] == 'eval'], 'rule_ids': order,
            'methods': {METHODS[k]: k for k in ('l12_frozen', 'minilm_lda', 'minilm_maxsim_loo', 'minilm_maxsim_self')}})
        self.write('artifacts/frozen/routing/published_metrics.json', self.read_run(run, 'metrics.json'))
        for source_method in ('minilm_lda', 'minilm_maxsim_loo', 'minilm_maxsim_self'):
            self.write(f'artifacts/frozen/routing/states_{METHODS[source_method]}.json',
                       self.read_run(run, f'states_{source_method}.json'))
        for rel in ('null_minilm.json', 'l12_reproduction.json', 'independent_audit.json'):
            self.write('artifacts/frozen/routing/' + rel, self.read_run(run, rel))
        for name, rel in [('without_eligibility', 'replay/n82_s42/cpu_s42.json'),
                          ('with_eligibility', 'replay/e8_gate/n82_s42/cpu_s42.json'),
                          ('alternate_order', 'replay/n82_s2026/cpu_s2026.json')]:
            obj = self.read(self.results_path(rel))
            assert len(obj['steps']) == 492
            self.write(f'artifacts/frozen/recognition/replay_{name}.json', obj)
            if name == 'without_eligibility':
                self.write('data/protocols/recognition_replay.json', {
                    'protocol': 'known_collection_evidence_replay',
                    'seed': obj['seed'], 'full_contrastive_bank_known_before_replay': True,
                    'snapshots': [{k: s[k] for k in ('event_index', 'kind', 'rule_id', 'k', 'written')}
                                  for s in obj['steps']]})

    def export_features(self, order):
        """Optional cached arrays, never loaded by the text-only runtime API."""
        import numpy as np

        def array(path):
            return np.load(io.BytesIO(self.raw(path)), allow_pickle=False)

        def save(rel, **arrays):
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            self.write_bytes(rel, buf.getvalue())

        corpus = self.read_run('e30_minilm_system_v1', 'corpus.json')
        cache = {}
        vectors = []
        for r in corpus:
            if r['split'] == 'fit':
                suffix = {'positive': 'pos', 'hard_negative': 'hn'}.get(r['kind'])
                path = self.results_path('evidence_h') / (f"{r['rule_id']}_{suffix}.npy" if suffix else 'calib_bg.npy')
                index = r['index']
            else:
                old = r['feature_path']
                marker = '/SIREN-TTCL-results/'
                if marker not in old:
                    raise ValueError('unknown archived feature identity')
                path = self.results / old.split(marker, 1)[1]
                index = r['feature_row']
            if path not in cache:
                cache[path] = array(path)
            vectors.append(cache[path][index])
        ids = np.asarray([r['uid'] for r in corpus], dtype=str)
        save('artifacts/frozen/features/llama_layer12.npz', ids=ids, features=np.stack(vectors))
        run = self.results_path('e30_minilm_system_v1')
        index = self.read(run / 'embedding_index.json')['row_indices']
        save('artifacts/frozen/features/minilm.npz', ids=ids, features=array(run / 'embeddings.npy')[index])
        self.write('artifacts/frozen/features/index.json', {
            'sample_ids': [r['uid'] for r in corpus], 'rows': len(corpus),
            'llama_layer12_shape': [len(corpus), 4096], 'minilm_shape': [len(corpus), 384],
            'purpose': 'Optional CPU reproduction cache; fit/eval metadata remains separate.',
            'llama_extraction': 'Block 12, mean over the last 60 percent of stream tokens; original archived precision.',
        })
        replay = self.read(self.results_path('replay/n82_s42/cpu_s42.json'))
        key_dir = self.results_path('replay/n82_s42/keys')
        final_dir = key_dir / replay['steps'][-1]['tag']
        final_keys = np.stack([array(final_dir / f'{r}_key.npy') for r in order])
        initial_steps = {s['rule_id']: s for s in replay['steps'] if s['k'] == 0}
        initial_keys = np.stack([array(key_dir / initial_steps[r]['tag'] / f'{r}_key.npy') for r in order])
        null_background = array(self.results / 'e1b/fit_h/calib_bg.npy')
        null = null_background.mean(axis=0).astype(np.float64)
        null /= np.linalg.norm(null)
        save('artifacts/frozen/routing/siren_keys.npz', initial_keys=initial_keys, final_keys=final_keys,
             null_prototype=null, null_construction_background=null_background,
             calibration_background=array(self.results_path('evidence_h/calib_bg.npy')))
        self.write('artifacts/frozen/routing/siren_null.json', {
            'key': null.tolist(), 'mean': replay['null_mu'], 'std': replay['null_sigma']})
        self.write('artifacts/frozen/routing/states_siren.json', {
            'rule_ids': order, 'null_mu': replay['null_mu'], 'null_sigma': replay['null_sigma'],
            'initial': {r: self.read(key_dir / initial_steps[r]['tag'] / f'{r}_meta.json') for r in order},
            'final': {r: self.read(final_dir / f'{r}_meta.json') for r in order},
        })
        # Execution uses a separately recorded fresh-float32 selector. Keep it
        # distinct from cached recognition features instead of conflating them.
        self.write('artifacts/frozen/routing/execution_selector.json',
                   self.read_run('e24_sequential_r13_v1', 'input/router.json'))

    def export_execution(self):
        e24 = 'e24_sequential_r13_v1'
        e26 = 'e26_pi_template_control_v1'
        outputs, packets, stages, cases = {}, {}, [], []
        labels = {run: self.read_run(run, 'judge/labels.json') for run in (e24, e26)}
        label_names = {'action': 'action', 'positive': 'positive', 'repetition': 'repetition',
                       'planning_only': 'planning_only', 'other_nonpositive': 'other_nonpositive',
                       'vote_count': 'vote_count', 'initial_disagreement': 'initial_disagreement'}

        def output(run, key):
            oid = ('shared:' if run == e24 else 'pi_v2:') + key
            if oid in outputs:
                return oid
            path = self.results_path(run) / 'generations' / (key + '.json')
            obj = self.read(path)
            task = obj['task']
            lab = labels[run][key]
            packet = lab['task_id']
            packet_path = self.results_path(run) / 'judge/cache' / packet / 'final.json'
            if packet not in packets:
                raw_packet = self.read(packet_path)
                packets[packet] = {'judge_packet_id': packet, **raw_packet}
            method = 'pi_v2' if run == e26 else METHODS[task['kind']]
            outputs[oid] = {
                'output_id': oid, 'method': method, 'sample_id': task['sample_id'],
                'selected_rule': task.get('selected_rule'), 'text': obj['text'],
                'generated_token_ids': obj['generated_token_ids'],
                'generated_tokens': obj['generated_tokens'], 'hit_cap': obj['hit_cap'],
                'labels': {dest: lab[src] for src, dest in label_names.items()},
                'judge_packet_id': packet, 'source_generation_sha256': self.sources[self.source_id(path)]['sha256'],
                'generation': task['generation'], 'precision': task['precision'],
                'model': {k: task['model'][k] for k in ('backbone', 'revision')},
                'prompt_record': obj.get('prompt_record', obj.get('prompt')),
            }
            return oid

        for phase, method, dirname in [('numeric', 'siren_kv', 'stages'), ('pi', 'pi_v1', 'pi_stages')]:
            oracle = {}
            for n in range(83):
                stage = self.read_run(e24, f'{dirname}/{n:02}.json')
                system = {sid: output(e24, v['task_id']) for sid, v in stage['system'].items()}
                current_oracle = {sid: output(e24, key) for sid, key in stage['oracle'].items()}
                oracle.update(current_oracle)
                stages.append({'method': method, 'installed': n, 'system': system,
                               'routes': {sid: v['route'] for sid, v in stage['system'].items()},
                               'oracle': current_oracle})
                if n == 0 and phase == 'numeric':
                    cases.extend({'sample_id': sid, 'method': 'base', 'mode': 'base', 'output_id': oid}
                                 for sid, oid in system.items())
                if n == 82:
                    cases.extend({'sample_id': sid, 'method': method, 'mode': 'system', 'output_id': oid}
                                 for sid, oid in system.items())
            assert len(oracle) == 666
            cases.extend({'sample_id': sid, 'method': method, 'mode': 'correct_rule', 'output_id': oid}
                         for sid, oid in oracle.items())
        pairs = self.read_run(e26, 'expression/input/pairs.json')
        oracle = {r['id']: output(e26, Path(r['provenance']['pi']['path']).stem) for r in pairs}
        # Generation file keys, not task IDs, identify the PI-v2 persisted output.
        for n in range(83):
            stage = self.read_run(e26, f'mappings/{n:02}.json')
            system = {sid: output(e24 if v['source'] == 'e24_base' else e26, v['key'])
                      for sid, v in stage['system'].items()}
            stages.append({'method': 'pi_v2', 'installed': n, 'system': system,
                           'routes': {sid: {'selected_rule': v['selected_rule']} for sid, v in stage['system'].items()},
                           'oracle': {}})
            if n == 82:
                cases.extend({'sample_id': sid, 'method': 'pi_v2', 'mode': 'system', 'output_id': oid}
                             for sid, oid in system.items())
        cases.extend({'sample_id': sid, 'method': 'pi_v2', 'mode': 'correct_rule', 'output_id': oid}
                     for sid, oid in oracle.items())
        self.write('artifacts/frozen/execution/outputs.jsonl', sorted(outputs.values(), key=lambda r: r['output_id']), lines=True)
        self.write('artifacts/frozen/execution/judge_packets.jsonl', sorted(packets.values(), key=lambda r: r['judge_packet_id']), lines=True)
        self.write('artifacts/frozen/execution/cases.jsonl', cases, lines=True)
        self.write('artifacts/frozen/execution/stages.jsonl', stages, lines=True)
        return {'outputs': len(outputs), 'judge_packets': len(packets), 'cases': len(cases), 'stage_rows': len(stages)}

    def export_expression(self):
        for method, run, prefix in [('pi_v1', 'e25_expression_analysis_v1', ''),
                                    ('pi_v2', 'e26_pi_template_control_v1', 'expression/')]:
            root = self.results_path(run) / prefix
            pairs = self.read(root / 'input/pairs.json')
            records, votes = [], {}
            for pair in pairs:
                result = self.read(root / 'results' / (pair['id'] + '.json'))
                for replicate, req in enumerate(result['requests'], 1):
                    if req in votes:
                        continue
                    p = root / 'cache' / req / 'vote.json'
                    identity = self.read(root / 'cache' / req / 'identity.json')
                    # A/B orientation is recoverable from the actual payload,
                    # including the rare case of equal replies (tie allowed).
                    votes[req] = {'request_id': req, 'replicate': replicate,
                                  'identity': identity, 'vote': self.read(p)}
                preference = result['merged']['preference']
                preference = {'numeric': 'siren_kv', 'pi': method}.get(preference, preference)
                records.append({'sample_id': pair['id'], 'rule_id': pair['rule_id'],
                                'split': GROUPS[pair['group']], 'correct_fired': pair['correct_fired'],
                                'outcome': pair['outcome'], 'preference': preference,
                                'comparison': ['siren_kv', method], 'pair': pair, 'result': result})
            self.write(f'artifacts/frozen/expression/{method}_pairs.jsonl', records, lines=True)
            self.write(f'artifacts/frozen/expression/{method}_votes.jsonl', list(votes.values()), lines=True)
            self.write(f'artifacts/frozen/expression/{method}_observations.json', self.read(root / 'input/observations.json'))
            self.write(f'artifacts/frozen/expression/{method}_published_analysis.json', self.read(root / 'analysis/results.json'))

    def export_prompts_and_supplement(self):
        for src, dest in [('configs/e19_judge_prompt_v3.txt', 'prompts/judge_behavior.txt'),
                          ('configs/e25/prompt_v1.txt', 'prompts/judge_expression.txt'),
                          ('configs/e25/rubric_v1.json', 'prompts/judge_expression_rubric.json'),
                          ('configs/e25/judge_v1.json', 'prompts/judge_expression_config.json')]:
            raw = self.raw(self.repo / src)
            if src.endswith('.json'):
                self.write(dest, json.loads(raw))
            else:
                self.write_bytes(dest, raw)
        config = self.read_run('e24_sequential_r13_v1', 'config.json')
        judge = {k: v for k, v in config['judge'].items() if k not in ('runtime_location', 'credential_env')}
        self.write('prompts/judge_behavior_config.json', judge)
        self.write('prompts/judge_behavior_calibration.json', self.read_run('e24_sequential_r13_v1', 'input/judge_freeze.json'))
        supplemental = {
            'legacy_instance_curves': ('results', 'e4/curves/instance_curves.json'),
            'legacy_maxsim': ('results', 'e4/arm5/arm5.json'),
            'legacy_rule_text_auc': ('results', 'e0b/auc_tables.json'),
            'legacy_fit_pool': ('repo', 'data/e4/evidence_v0.json'),
            'legacy_supervision': ('results', 'siren100/e29_lcfm_revision_v1r1/supervision.json'),
            'legacy_matched_baseline': ('results', 'siren100/e29_lcfm_revision_v1r1/baseline.json'),
            'legacy_matched_audit': ('results', 'siren100/e29_lcfm_revision_v1r1/audit.json'),
            'value_binary_pos': ('results', 'siren100/e10_selectivity/binary_pos.json'),
            'value_binary_neg': ('results', 'siren100/e10_selectivity/binary_neg.json'),
            'value_binary_summary': ('results', 'siren100/e10_selectivity/summary.json'),
            'value_neg_generated': ('results', 'siren100/e10_selectivity/gen_neg_edit.json'),
            'base_neg_generated': ('results', 'siren100/e10_selectivity/gen_neg_base.json'),
            'value_admission': ('results', 'siren100/e5_react/gpu_n82/e5_react_train.json'),
            'value_original_preferences': ('results', 'siren100/e12_judge/judged_OPUS.rows.jsonl'),
            'value_turn_removed_preferences': ('results', 'siren100/e13_judge/judged_E13.rows.jsonl'),
            'value_original_generations': ('results', 'siren100/e5_react/pilot_n82_raw.json'),
            'value_turn_removed_generations': ('results', 'siren100/e13_silver_v1/gen_silver_edit.json'),
            'base_turn_removed_generations': ('results', 'siren100/e13_silver_v1/gen_silver_base.json'),
            'value_turn_removed_oracle_generations': ('results', 'siren100/e14_oracle/gen_oracle.json'),
            'value_distillation': ('results', 'siren100/e15_mini_rerun_v1/report/stats_final.json'),
            'value_distillation_labels': ('results', 'siren100/e15_mini_rerun_v1/judge/merged_final.json'),
            'value_distillation_pairmap': ('results', 'siren100/e15_mini_rerun_v1/judge/private_pairmap_final.json'),
            'original_route': ('repo', 'docs/e11/silver_route_prediction.json'),
            'turn_removed_route': ('repo', 'docs/e13/silver_v1_route_prediction.json'),
            'new_context_route': ('repo', 'docs/e14/silver_v2_route_prediction.json'),
            'turn_removed_manifest': ('repo', 'docs/e13/silver_v1_manifest.json'),
            'new_context_manifest': ('repo', 'docs/e14/silver_v2_manifest.json'),
            'value_original_summary': ('repo', 'docs/e12/stats_summary.json'),
            'value_turn_removed_summary': ('repo', 'docs/e13/stats_summary.json'),
            'value_view_comparison': ('repo', 'docs/e13/compare_v0_v1.json'),
            'value_oracle_comparison': ('repo', 'docs/e14/stats_r1.json'),
        }
        for name, (location, rel) in supplemental.items():
            path = (self.repo if location == 'repo' else self.results) / rel
            if not path.exists():
                self.missing.append(name)
                continue
            if path.suffix == '.jsonl':
                rows = [json.loads(line) for line in self.raw(path).splitlines() if line.strip()]
                self.write(f'artifacts/frozen/supplement/{name}.jsonl', rows, lines=True)
            else:
                self.write(f'artifacts/frozen/supplement/{name}.json', self.read(path))

    def finish(self, counts):
        self.write('artifacts/source_map.json', {
            'schema': 'siren-paper-export-v1', 'paper_version': 'LCFM camera-ready v1.6',
            'source_identifiers': 'Relative archive identities are provenance, not required private paths.',
            'methods': METHODS, 'sources': self.sources, 'exports': self.exports,
            'missing_optional_sources': self.missing,
        })
        self.write('data/export_audit.json', {
            'schema': 'siren-data-export-audit-v1', 'rules': 82,
            'recognition': {'fit': 3072, 'evaluation': 2390},
            'execution': {'fit': 491, 'validation': 164, 'development': 112, 'main': 464, 'seed': 90},
            'evidence': counts, 'missing_optional_sources': self.missing,
            'source_map_sha256': self.exports['artifacts/source_map.json']['sha256'],
            'no_model_api_calls': True, 'no_training': True,
        })
        # Includes every byte in the export; the manifest does not hash itself.
        self.write('artifacts/frozen/manifest.json', {'schema': 'siren-frozen-assets-v1', 'files': dict(self.exports)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-repo', type=Path, required=True)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    exporter = Exporter(args.source_repo, args.results_root, args.output_root)
    # Remove names emitted by the pre-release exporter only. Never prune the
    # author's archive or unrelated public files.
    for stem in ('predictions_minilm_maxsim_self_calibrated.jsonl',
                 'scores_minilm_maxsim_self_calibrated.npz',
                 'states_minilm_maxsim_self_calibrated.json'):
        (exporter.out / 'artifacts/frozen/routing' / stem).unlink(missing_ok=True)
    order = exporter.export_data()
    exporter.export_routing(order)
    exporter.export_features(order)
    counts = exporter.export_execution()
    exporter.export_expression()
    exporter.export_prompts_and_supplement()
    exporter.finish(counts)
    print(json.dumps({'complete': True, **counts, 'missing_optional_sources': exporter.missing}, sort_keys=True))


if __name__ == '__main__':
    main()
