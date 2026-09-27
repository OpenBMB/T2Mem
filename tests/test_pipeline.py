"""Offline integration checks for the real CLI runners and OpenAI SDK.

The server and sentence-transformers stub are deterministic test doubles. These
checks validate wiring and schemas, not benchmark accuracy or embedding quality.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MockEndpoint(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        if self.path.endswith('/embeddings'):
            payload = {'object': 'list', 'data': [
                {'object': 'embedding', 'index': i, 'embedding': [1.0, 0.0]}
                for i, _ in enumerate(body['input'])
            ], 'model': body['model'], 'usage': {'prompt_tokens': 10, 'total_tokens': 10}}
        else:
            model = body['model']
            if model == 'test-index':
                content = json.dumps({'summary': 'Alice lives in Paris.',
                                      'atomic_facts': ['Alice lives in Paris.']})
            elif model == 'test-summary':
                content = 'Alice lives in Paris.'
            elif model == 'test-controller':
                action = self.server.actions.pop(0) if self.server.actions else 'ANSWER'
                content = json.dumps({'action': action, 'reasoning': 'Found evidence.',
                                      'target_wiki_id': [self.server.target_wiki],
                                      'evidence': [{'snippet': 'Alice lives in Paris.',
                                                    'date_time': '2023-05-01'}]})
            elif model == 'test-invalid-controller':
                content = json.dumps({'action': 'INVALID', 'evidence': []})
            elif model == 'test-broken-controller':
                content = 'not a JSON response'
            elif model == 'test-answer':
                content = 'Paris'
            elif model == 'test-judge':
                prompt = body['messages'][-1]['content']
                content = json.dumps({'label': 'CORRECT'}) if "'CORRECT'" in prompt else 'yes'
            else:
                raise AssertionError(f'Unexpected model {model}')
            payload = {'id': 'test', 'object': 'chat.completion', 'created': 0,
                       'model': model,
                       'choices': [{'index': 0, 'finish_reason': 'stop',
                                    'message': {'role': 'assistant', 'content': content}}],
                       'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), MockEndpoint)
        cls.server.requests = []
        cls.server.actions = []
        cls.server.target_wiki = ''
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f'http://127.0.0.1:{cls.server.server_port}/v1'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='t2mem-test-')
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.data = self.work / 'data'
        (self.data / 'raw').mkdir(parents=True)
        # Only test doubles are placed on PYTHONPATH; no model download or torch.
        stubs = self.work / 'stubs'
        stubs.mkdir()
        (stubs / 'sentence_transformers.py').write_text(
            'import numpy as np\n'
            'class SentenceTransformer:\n'
            '    def __init__(self, *args, **kwargs): pass\n'
            '    def encode(self, texts, **kwargs):\n'
            '        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)\n'
        )
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('T2MEM_', 'JUDGE_'))}
        for key in ('VARIANT', 'DATA', 'WIKI', 'WIKI_ROOT', 'RETRIEVAL', 'OUT', 'EMBED_BACKEND',
                    'MODEL', 'BASE_URL', 'API_KEY', 'RAW_DATA', 'DATA_ROOT', 'OUT_DIR',
                    'EMBED_MODEL_PATH', 'EMBED_BASE_URL', 'EMBED_MODEL', 'EMBED_API_KEY',
                    'OPENAI_API_KEY', 'OPENAI_BASE_URL'):
            self.env.pop(key, None)
        self.env.update(PYTHON=sys.executable, PYTHONPATH=str(stubs), DATA_ROOT=str(self.data),
                        T2MEM_LLM_BASE_URL=self.base_url, T2MEM_LLM_MODEL_NAME='test-index',
                        T2MEM_EMBEDDING_BASE_URL=self.base_url, T2MEM_EMBEDDING_MODEL_NAME='test-embed',
                        JUDGE_BASE_URL=self.base_url, JUDGE_API_KEY='test-key', JUDGE_MODEL='test-judge',
                        WORKERS='1', TOP_K='1', MAX_STEPS='5', OUT_DIR=str(self.work / 'eval'))
        self.server.requests.clear()

    def run_script(self, name, *args, env=None, success=True):
        result = subprocess.run(['bash', str(ROOT / 'scripts' / name), *map(str, args)],
                                cwd=self.work, env={**self.env, **(env or {})},
                                capture_output=True, text=True, timeout=30)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def write_raw(self, dataset, direct=False):
        if dataset == 'locomo':
            raw = [{'sample_id': 'test-conv', 'conversation': {
                'session_1_date_time': '2023-05-01',
                'session_1': [{'speaker': 'Alice', 'dia_id': 'D1:1', 'text': 'I live in Paris. 测试'}],
                'session_2_date_time': '2023-05-02',
                'session_2': [{'speaker': 'Alice', 'dia_id': 'D2:1', 'text': 'I visited Rome.'}]},
                'qa': [{'question': 'Where does Alice live?', 'answer': 'Paris', 'category': 1},
                       {'question': 'Unanswerable?', 'answer': '', 'category': 5}]}]
            raw_name = 'locomo10.json'
            self.server.target_wiki = 'test-conv_session_1'
        else:
            raw = [{'question_id': 'test-q', 'question_type': 'single-session-user',
                    'question': 'Where do I live?', 'answer': 'Paris', 'question_date': '2023-05-03',
                    'answer_session_ids': ['s1'], 'haystack_session_ids': ['s1', 's2'],
                    'haystack_dates': ['2023-05-01', '2023-05-02'],
                    'haystack_sessions': [[{'role': 'user', 'content': 'I live in Paris. 测试'}],
                                          [{'role': 'user', 'content': 'I visited Rome.'}]]}]
            raw_name = 'longmemeval_s_cleaned.json'
            self.server.target_wiki = 'test-q_session_1'
        raw_path = (self.data if direct else self.data / 'raw') / raw_name
        raw_path.write_text(json.dumps(raw))
        return raw_path

    def prepare(self, dataset, variant='structured'):
        self.write_raw(dataset)
        self.env['VARIANT'] = variant
        self.run_script('preprocess_data.sh', dataset)
        model = 'test-summary' if variant == 'summary' else 'test-index'
        self.run_script('run_build_wiki.sh', dataset, env={'MODEL': model})
        suffix = 'summary_wiki' if variant == 'summary' else 'wiki'
        wiki = self.data / dataset / f'{dataset}_{suffix}.jsonl'
        rows = [json.loads(line) for line in wiki.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.get('summary') and not row.get('error') for row in rows))
        self.run_script('run_retrieval.sh', dataset, 'wiki')
        return wiki

    def test_structured_modes_and_evaluation(self):
        for dataset in ('locomo', 'longmemeval'):
            with self.subTest(dataset=dataset):
                self.prepare(dataset)
                # Retain two retrieved entries, but initially inject one.
                self.run_script('run_retrieval.sh', dataset, 'wiki', env={'TOP_K': '2'})
                self.run_script('run_retrieval.sh', dataset, 'raw', env={'TOP_K': '2'})
                for mode in ('full', 'wiki-only', 'raw-only'):
                    with self.subTest(mode=mode):
                        self.server.actions = ({'full': ['READ_RAW_FROM_WIKI', 'SEARCH_WIKI', 'ANSWER'],
                                                'wiki-only': ['SEARCH_WIKI', 'ANSWER'],
                                                'raw-only': ['SEARCH_RAW', 'ANSWER']}[mode]).copy()
                        pred = self.work / f'{dataset}_{mode}.jsonl'
                        self.run_script('run_t2mem.sh', dataset, mode,
                                        '--controller-model-name', 'test-controller',
                                        '--answer-model-name', 'test-answer', env={'OUT': str(pred)})
                        rows = [json.loads(line) for line in pred.read_text().splitlines()]
                        self.assertEqual(len(rows), 1)  # LoCoMo category 5 excluded.
                        row = rows[0]
                        self.assertFalse(row.get('error'), row)
                        self.assertEqual(row['prediction'], 'Paris')
                        self.assertEqual(row['stop_reason'], 'ANSWER')
                        self.assertEqual(len(row['accumulated_evidence']), 1)
                        self.assertEqual(row['controller_steps'][0]['action'],
                                         {'full': 'READ_RAW_FROM_WIKI', 'wiki-only': 'SEARCH_WIKI',
                                          'raw-only': 'SEARCH_RAW'}[mode])
                        if dataset == 'locomo':
                            usage = pred.with_name(f'{pred.stem}_token_usage_summary.json')
                            self.assertTrue(usage.is_file())
                        self.run_script('run_eval.sh', dataset, pred)
                        summary = json.loads((self.work / 'eval' / f'judge_{pred.stem}.summary.json').read_text())
                        self.assertEqual(summary['accuracy'], 1.0)

    def test_summary_variants(self):
        for dataset in ('locomo', 'longmemeval'):
            with self.subTest(dataset=dataset):
                self.prepare(dataset, 'summary')
                self.run_script('run_retrieval.sh', dataset, 'wiki', env={'TOP_K': '2'})
                self.server.actions = ['ANSWER']
                pred = self.work / f'{dataset}_summary.jsonl'
                self.run_script('run_t2mem.sh', dataset, 'full', '--top-k', '2',
                                '--controller-model-name', 'test-controller',
                                '--answer-model-name', 'test-answer', env={'OUT': str(pred)})
                row = json.loads(pred.read_text())
                self.assertFalse(row.get('error'), row)
                self.assertEqual(row['prediction'], 'Paris')
                self.assertEqual(len(row['initial_prediction_wiki_context']), 2)
                self.run_script('run_eval.sh', dataset, pred)

    def test_invalid_action_and_failed_prediction_retention(self):
        for dataset in ('locomo', 'longmemeval'):
            with self.subTest(dataset=dataset):
                self.prepare(dataset)
                pred = self.work / f'{dataset}_invalid.jsonl'
                self.run_script('run_t2mem.sh', dataset, 'wiki-only',
                                '--controller-model-name', 'test-invalid-controller',
                                '--answer-model-name', 'test-answer', env={'OUT': str(pred)})
                row = json.loads(pred.read_text())
                self.assertEqual(row['action_fallback_count'], 1)
                self.assertEqual(row['stop_reason'], 'ANSWER')
                self.run_script('run_t2mem.sh', dataset, 'full', '--max-retries', '1',
                                '--controller-model-name', 'test-broken-controller',
                                '--answer-model-name', 'test-answer', env={'OUT': str(pred)})
                row = json.loads(pred.read_text())
                self.assertEqual(row['stop_reason'], 'ERROR')
                self.assertTrue(row['error'])
                self.assertEqual(row['prediction'], '')
                if dataset == 'locomo':
                    self.run_script('run_eval.sh', dataset, pred)
                    adapter = self.work / 'eval' / f'judge_{pred.stem}_input.json'
                    self.assertEqual(len(json.loads(adapter.read_text())['test-conv']), 1)

    def test_locomo_budget_and_prompt_dump(self):
        self.prepare('locomo')
        pred = self.work / 'budget.jsonl'
        self.run_script('run_t2mem.sh', 'locomo', 'wiki-only', '--output-budget-tokens', '10',
                        '--dump-prompts', '--controller-model-name', 'test-controller',
                        '--answer-model-name', 'test-answer', env={'OUT': str(pred)})
        row = json.loads(pred.read_text())
        self.assertEqual(row['stop_reason'], 'output_budget_exhausted')
        self.assertEqual(row['controller_calls'], 0)
        self.assertTrue(row['accumulated_evidence'])
        self.assertIn('answer_user_prompt', row)

    def test_invalid_runner_combinations(self):
        for script, args, env, message in (
            ('run_build_wiki.sh', ['invalid'], {}, 'unknown dataset'),
            ('run_build_wiki.sh', ['locomo'], {'VARIANT': 'invalid'}, 'unknown VARIANT'),
            ('run_retrieval.sh', ['locomo', 'invalid'], {}, 'unknown unit'),
            ('run_retrieval.sh', ['locomo', 'raw'], {'VARIANT': 'summary'}, 'summary retrieval uses wiki'),
            ('run_retrieval.sh', ['longmemeval', 'wiki'], {'EMBED_BACKEND': 'invalid'}, 'unknown EMBED_BACKEND'),
            ('run_t2mem.sh', ['locomo', 'wiki-only'], {'VARIANT': 'summary'}, 'full mode only'),
        ):
            with self.subTest(script=script, args=args, env=env):
                result = self.run_script(script, *args, env=env, success=False)
                self.assertIn(message, result.stderr)


    def run_entry(self, *args, env=None, success=True):
        config = self.work / '.env'
        config.write_text('\n'.join([
            f'T2MEM_LLM_BASE_URL="{self.base_url}"',
            'T2MEM_LLM_MODEL_NAME="test-index"',
            'T2MEM_CONTROLLER_MODEL_NAME="test-controller"',
            'T2MEM_ANSWER_MODEL_NAME="test-answer"',
            f'T2MEM_EMBEDDING_BASE_URL="{self.base_url}"',
            'T2MEM_EMBEDDING_MODEL_NAME="test-embed"',
            f'JUDGE_BASE_URL="{self.base_url}"',
            'JUDGE_API_KEY="test-key"',
            'JUDGE_MODEL="test-judge"',
        ]) + '\n')
        process_env = {k: v for k, v in self.env.items()
                       if not k.startswith(('T2MEM_', 'JUDGE_'))}
        process_env.update(env or {})
        result = subprocess.run([
            sys.executable, str(ROOT / 'run.py'), '--env-file', str(config),
            '--data-dir', str(self.data), '--output-dir', str(self.work / 'outputs'),
            '--eval-dir', str(self.work / 'eval'), *map(str, args),
        ], cwd=self.work, env=process_env, capture_output=True, text=True, timeout=45)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_one_command_discovers_both_datasets_and_loads_dotenv(self):
        originals = {self.write_raw(dataset, direct=True): None
                     for dataset in ('locomo', 'longmemeval')}
        originals = {path: path.read_bytes() for path in originals}
        self.server.actions = []
        self.run_entry('--check')
        self.assertEqual(self.server.requests, [])
        self.run_entry('--temperature', '0.0', '--answer-max-tokens', '1024')
        generation_requests = [request for request in self.server.requests if 'messages' in request]
        self.assertTrue(generation_requests)
        self.assertTrue(all(request['temperature'] == 0.0 for request in generation_requests))
        for dataset in ('locomo', 'longmemeval'):
            pred = self.work / 'outputs' / f'{dataset}_structured_t2mem_full_answers.jsonl'
            row = json.loads(pred.read_text())
            self.assertEqual(row['prediction'], 'Paris')
            summary = self.work / 'eval' / f'judge_{pred.stem}.summary.json'
            self.assertEqual(json.loads(summary.read_text())['accuracy'], 1.0)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    def test_individual_stages_and_all_api_representations(self):
        for dataset in ('locomo', 'longmemeval'):
            self.write_raw(dataset, direct=True)
            self.run_entry(dataset, '--stage', 'preprocess')
            self.run_entry(dataset, '--stage', 'index')
            self.run_entry(dataset, '--stage', 'retrieve', '--read-mode', 'raw-only')
            self.run_entry(dataset, '--stage', 'infer', '--read-mode', 'raw-only')
            self.run_entry(dataset, '--stage', 'evaluate', '--read-mode', 'raw-only')
            # Summary retrieval is API-backed with only requirements.txt.
            self.run_entry(dataset, '--variant', 'summary', '--skip-eval')
            self.assertTrue((self.data / dataset / f'{dataset}_retrieval_summary_wiki.jsonl').is_file())

    def test_preflight_errors_and_failed_stage_stop(self):
        result = self.run_entry(success=False)
        self.assertIn('No dataset found', result.stderr)
        self.write_raw('locomo', direct=True)
        result = self.run_entry('locomo', env={'JUDGE_API_KEY': ''}, success=False)
        self.assertIn('JUDGE_API_KEY', result.stderr)
        self.assertEqual(self.server.requests, [])
        self.assertFalse((self.data / 'locomo_processed.json').exists())
        result = self.run_entry('locomo', '--temperature', 'nan', success=False)
        self.assertIn('nonnegative finite', result.stderr)
        self.assertEqual(self.server.requests, [])
        result = self.run_entry('locomo', '--check', env={'T2MEM_LLM_MODEL_NAME': 'your-served-model'}, success=False)
        self.assertIn('served model name', result.stderr)
        # Root and raw/ copies are ambiguous; explicit --data-path resolves it.
        raw = self.write_raw('locomo')
        result = self.run_entry('locomo', '--stage', 'preprocess', success=False)
        self.assertIn('Multiple locomo', result.stderr)
        self.run_entry('locomo', '--stage', 'preprocess', '--data-path', raw)
        self.run_entry('locomo', '--stage', 'index')
        self.run_entry('locomo', '--stage', 'retrieve')
        # Unified flags reach the controller; failed inference is a stage failure.
        self.run_entry('locomo', '--stage', 'infer', '--output-budget-tokens', '10', '--dump-prompts')
        prediction = self.work / 'outputs/locomo_structured_t2mem_full_answers.jsonl'
        row = json.loads(prediction.read_text())
        self.assertEqual(row['controller_calls'], 0)
        self.assertIn('answer_user_prompt', row)
        result = self.run_entry('locomo', '--stage', 'infer',
                                '--controller-model-name', 'test-broken-controller', success=False)
        self.assertIn('rows failed', result.stderr)
        self.assertTrue(json.loads(prediction.read_text())['error'])
        wiki = self.data / 'locomo/locomo_wiki.jsonl'
        wiki.write_text(json.dumps({'wiki_id': 'bad', 'error': 'fixture failure'}) + '\n')
        requests_before = len(self.server.requests)
        result = self.run_entry('locomo', '--stage', 'retrieve', success=False)
        self.assertIn('rows failed', result.stderr)
        self.assertEqual(len(self.server.requests), requests_before)

    def test_cli_help_and_shell_syntax(self):
        skip = {'__init__.py', 'config.py', 'embeddings.py', 'pipeline.py', 'ada_reading.py', 't2mem_prompt.py',
                'llm_utils.py', 'read_locomo.py', 'read_longmemeval.py'}
        for path in sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'evaluation').glob('*.py')):
            if path.name not in skip:
                result = subprocess.run([sys.executable, str(path), '--help'], cwd=self.work,
                                        env=self.env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, f'{path}\n{result.stderr}')
        for path in (ROOT / 'scripts').glob('*.sh'):
            subprocess.run(['bash', '-n', str(path)], check=True)


if __name__ == '__main__':
    unittest.main()
