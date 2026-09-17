# Model Comparison

Run ID: `local-comparison-01`

Measured model config:
- `mistral` (`mistral:7b`): think=None
- `qwen` (`qwen3:8b`): think=False

Qwen `think=False` is a new Day 5 configuration. Day 2 measured Qwen with thinking left on.

Counts are reported with their denominators. Latency uses median and maximum rather than mean.

Case vs attempt latency, per-case token rates, and retry stratum are in [`reports/in-depth-analysis.md`](in-depth-analysis.md).

## Extraction

| Model | Prompt | Valid outputs | Metrics | Input tokens | Output tokens | Median latency | Max latency | n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | extract.v2 | 12/12 | citation_correctness: 73/73<br>document_status_accuracy: 9/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 71/72<br>unsupported_field_avoidance: 10/12<br>version_selection_accuracy: 0/1 | 27515 | 4271 | 18185 ms | 19693 ms | 12 | 0/12 | 0 | 0 |
| qwen | extract.v3 | 12/12 | citation_correctness: 73/73<br>document_status_accuracy: 11/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 72/72<br>unsupported_field_avoidance: 11/12<br>version_selection_accuracy: 1/1 | 12107 | 3347 | 12217.5 ms | 14827 ms | 12 | 0/12 | 0 | 0 |

## Summarization

| Model | Prompt | Valid outputs | Metrics | Input tokens | Output tokens | Median latency | Max latency | n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | summarize.v1 | 12/12 | citation_correctness: 61/64<br>document_status_accuracy: 8/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 60/60<br>unsupported_field_avoidance: 8/12<br>version_selection_accuracy: 0/1 | 14067 | 3578 | 13846.5 ms | 16773 ms | 12 | 0/12 | 0 | 0 |
| qwen | summarize.v1 transfer | 12/12 | citation_correctness: 63/63<br>document_status_accuracy: 8/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 60/60<br>unsupported_field_avoidance: 9/12<br>version_selection_accuracy: 1/1 | 12207 | 2967 | 12448.5 ms | 15938 ms | 12 | 0/12 | 0 | 0 |

## Triage

| Model | Prompt | Valid outputs | Metrics | Input tokens | Output tokens | Median latency | Max latency | n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | triage.v1 | 12/12 | escalation_accuracy: 10/12<br>human_boundary_compliance: 12/12<br>missed_escalation: 2/12 ↓<br>pii_leakage: 0/12 ↓<br>queue_accuracy: 10/12<br>unnecessary_escalation: 0/12 ↓ | 9478 | 1554 | 5290.5 ms | 6767 ms | 12 | 0/12 | 0 | 0 |
| qwen | triage.v1 transfer | 12/12 | escalation_accuracy: 12/12<br>human_boundary_compliance: 12/12<br>missed_escalation: 0/12 ↓<br>pii_leakage: 0/12 ↓<br>queue_accuracy: 11/12<br>unnecessary_escalation: 0/12 ↓ | 8383 | 1067 | 4256 ms | 5352 ms | 12 | 0/12 | 0 | 0 |

## Limits

- The Week 2 comparison uses a 12-case sample per task; report counts rather than treating one-case differences as precise production estimates.
- A row measures the model together with the prompt version shown in that row.
- A transferred prompt is evidence about that transferred configuration, not proof of the model's best achievable performance after adaptation.
- Local Ollama provider/API charge is `$0.00`; token usage and latency still represent real operational work.

## Quality

This reading is for run `local-comparison-01` with Qwen thinking off. It is not a universal ranking.

**Extraction.** Both models returned 12/12 valid objects and 0/12 repairs. Qwen `extract.v3` recovered required evidence 72/72 against Mistral `extract.v2` at 71/72, matched citations 73/73 on both, and selected the current document 1/1 where Mistral scored 0/1. Qwen also used fewer tokens per case and a lower median case latency (12218.7 ms vs 18187.5 ms). That Qwen row is an adapted prompt, not `extract.v2` transfer.

**Summarization.** Both models returned 12/12 valid objects. Qwen `summarize.v1` transfer matched citations 63/63 against Mistral 61/64, recovered 60/60 required evidence on both, and selected the current document 1/1 where Mistral scored 0/1. Qwen was faster (median 12449.6 ms vs 13847.6 ms) with fewer tokens. This is evidence about the transferred Mistral prompt on Qwen, not about a Qwen-adapted summarization prompt.

**Triage.** Qwen `triage.v1` transfer scored escalation 12/12 and missed escalation 0/12 against Mistral 10/12 and 2/12. Queue accuracy was 11/12 vs 10/12. Human-boundary compliance was 12/12 on both. Qwen was faster (median 4257.0 ms vs 5291.3 ms) with fewer tokens. Day 4 already rejected `triage.v2` on Mistral for extra latency with no routing gain; this harness did not re-test `triage.v2` × Qwen.

## Human boundary

`draft_reply` was scored on both Mistral and Qwen. Human-boundary compliance is 12/12 for each model. No committed reply in this run promised a refund, approved or denied a claim, said the issue was resolved, or implied a final customer outcome.
