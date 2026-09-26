from __future__ import annotations

import asyncio
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from uuid import uuid4

from prama.evaluator import get_cer, get_wer
from prama_server.evaluator import EvaluationInferenceResult
from prama_server.servicer import http


def rows():
    return [
        {"id": "speaker_1", "index": 1, "reference": "one two three", "hypothesis": "one three"},
        {"id": "speaker_2", "index": 2, "reference": "你好世界", "hypothesis": "你好世间"},
        {"id": "speaker_3", "index": 3, "reference": "", "hypothesis": "extra"},
        {"id": "speaker_4", "index": 4, "reference": "a b", "hypothesis": ""},
    ]


class AsrMetricStreamTest(unittest.TestCase):
    def test_streamed_counts_and_tokens_match_batch_across_threads(self):
        def evaluate(_):
            job = http.EvaluationJob(uuid4().hex, http.EvaluationRequest())
            for row in rows():
                http._publish_asr_metrics(job, row)
            for metric, fn in (("wer", get_wer), ("cer", get_cer)):
                expected = fn([r["reference"] for r in rows()],
                              [r["hypothesis"] for r in rows()],
                              [r["id"] for r in rows()])
                actual = job.result[f"{metric}_report"]
                self.assertEqual(actual["summary"], http._wer_counts_to_payload(expected.summary))
                self.assertEqual(
                    [u["tokens"] for u in actual["utterances"]],
                    [[{"label": t.eval_label, "ref": t.ref_word, "hyp": t.hyp_word}
                      for t in u.tokens] for u in expected.utterances],
                )
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(evaluate, range(32)))

    def test_emits_real_sse_before_next_inference_and_final_matches(self):
        job = http.EvaluationJob(uuid4().hex, http.EvaluationRequest())
        http.jobs[job.job_id] = job
        self.addCleanup(http.jobs.pop, job.job_id, None)
        http.message_manager.register_job(job.job_id)
        response = http.stream_evaluation_events(job.job_id)

        async def read_first_record():
            messages = []
            while len(messages) < 2:
                event = await asyncio.wait_for(anext(response.body_iterator), 5)
                if "event: metric_result" in event:
                    messages.append(event)
            for metric, event in zip(("wer", "cer"), messages):
                self.assertIn("event: metric_result", event)
                payload = json.loads(event.split("data: ", 1)[1])
                self.assertEqual(payload["metric"], metric)
                self.assertEqual(payload["utterance"]["id"], "speaker_1")
                self.assertEqual(payload["summary"]["sentence_count"], 1)
            self.assertEqual(job.status, "running")
            await response.body_iterator.aclose()

        def infer(_inferencer, on_infer_result, **_kwargs):
            for index, row in enumerate(rows()):
                on_infer_result(EvaluationInferenceResult(
                    tag=job.job_id, id=row["id"], reference=row["reference"],
                    hypothesis=row["hypothesis"],
                ))
                if index == 0:
                    asyncio.run(read_first_record())
            return http._build_asr_metrics(rows())

        with patch.object(http, "_load_evaluation_dataset", return_value=rows()), \
             patch.object(http, "_register_asr_sample_records"), \
             patch.object(http, "AsrGrpcInferencer"), \
             patch.object(http, "Evaluator") as evaluator:
            evaluator.return_value.__enter__.return_value.iter_evaluate.side_effect = infer
            http._run_evaluation(job)
        self.assertEqual(job.status, "completed", job.error)
        self.assertEqual(job.result["wer_report"]["summary"], http._build_wer_report(rows())["summary"])
        self.assertEqual(job.result["cer_report"]["summary"], http._build_cer_report(rows())["summary"])


if __name__ == "__main__":
    unittest.main()
