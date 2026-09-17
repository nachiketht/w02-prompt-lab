# Model Comparison

Run ID: `local-comparison-01`

Measured model config:
- `mistral` (`mistral:7b`): think=None
- `qwen` (`qwen3:8b`): think=False

Qwen `think=False` is a new Day 5 configuration. Day 2 measured Qwen with thinking left on.

Counts are reported with their denominators. Headline latency is median and maximum **case** end-to-end time (`elapsed_ms`), with one observation per case. Attempt latency is HTTP-call time and uses a separate `n`. Mean latency is not used.

Provider/API charge is `$0.00`.

## Extraction

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median case latency | Max case latency | Case n | Median attempt latency | Max attempt latency | Attempt n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | extract.v2 | 12/12 | citation_correctness: 73/73<br>document_status_accuracy: 9/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 71/72<br>unsupported_field_avoidance: 10/12<br>version_selection_accuracy: 0/1 | 2292.9 | 355.9 | 18187.5 ms | 19694.9 ms | 12 | 18185 ms | 19693 ms | 12 | 0/12 | 0 | 0 |
| qwen | extract.v3 | 12/12 | citation_correctness: 73/73<br>document_status_accuracy: 11/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 72/72<br>unsupported_field_avoidance: 11/12<br>version_selection_accuracy: 1/1 | 1008.9 | 278.9 | 12218.7 ms | 14828.4 ms | 12 | 12217.5 ms | 14827 ms | 12 | 0/12 | 0 | 0 |

## Summarization

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median case latency | Max case latency | Case n | Median attempt latency | Max attempt latency | Attempt n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | summarize.v1 | 12/12 | citation_correctness: 61/64<br>document_status_accuracy: 8/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 60/60<br>unsupported_field_avoidance: 8/12<br>version_selection_accuracy: 0/1 | 1172.2 | 298.2 | 13847.6 ms | 16776.0 ms | 12 | 13846.5 ms | 16773 ms | 12 | 0/12 | 0 | 0 |
| qwen | summarize.v1 transfer | 12/12 | citation_correctness: 63/63<br>document_status_accuracy: 8/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 60/60<br>unsupported_field_avoidance: 9/12<br>version_selection_accuracy: 1/1 | 1017.2 | 247.2 | 12449.6 ms | 15939.9 ms | 12 | 12448.5 ms | 15938 ms | 12 | 0/12 | 0 | 0 |

## Triage

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median case latency | Max case latency | Case n | Median attempt latency | Max attempt latency | Attempt n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | triage.v1 | 12/12 | escalation_accuracy: 10/12<br>human_boundary_compliance: 12/12<br>missed_escalation: 2/12 ↓<br>pii_leakage: 0/12 ↓<br>queue_accuracy: 10/12<br>unnecessary_escalation: 0/12 ↓ | 789.8 | 129.5 | 5291.3 ms | 6770.8 ms | 12 | 5290.5 ms | 6767 ms | 12 | 0/12 | 0 | 0 |
| qwen | triage.v1 transfer | 12/12 | escalation_accuracy: 12/12<br>human_boundary_compliance: 12/12<br>missed_escalation: 0/12 ↓<br>pii_leakage: 0/12 ↓<br>queue_accuracy: 11/12<br>unnecessary_escalation: 0/12 ↓ | 698.6 | 88.9 | 4257.0 ms | 5353.3 ms | 12 | 4256 ms | 5352 ms | 12 | 0/12 | 0 | 0 |

## Limits

- Each task uses 12 cases. Results are directional, not production-scale estimates.
- A row measures the model together with the prompt version shown in that row.
- Prompt-transfer rows are labeled `transfer`. They are evidence about that transferred configuration, not proof of the model's best performance after adaptation.
- Untested combinations in this harness include `triage.v2` × Qwen and any prompt version that does not appear in a table row.
- Case latency `n` is 12 (one observation per case). Attempt latency `n` is HTTP calls. Do not treat those as the same observation count.
- Retry/repair cases already have a larger case `elapsed_ms`. They are not weighted again into the median or max.
- Local Ollama latency depends on lab hardware. No production-volume reliability claim is being made.
- 11/12 versus 10/12 is not a universal model ranking.
- Local provider/API charge is `$0.00`; token usage and latency still represent real operational work.

## Quality

This reading is for run `local-comparison-01` with Qwen thinking off. It is not a universal ranking.

**Extraction.** Both models returned 12/12 valid objects and 0/12 repairs. Qwen `extract.v3` recovered required evidence 72/72 against Mistral `extract.v2` at 71/72, matched citations 73/73 on both, and selected the current document 1/1 where Mistral scored 0/1. Qwen also used fewer tokens per case and a lower median case latency (12218.7 ms vs 18187.5 ms). That Qwen row is an adapted prompt, not `extract.v2` transfer.

**Summarization.** Both models returned 12/12 valid objects. Qwen `summarize.v1` transfer matched citations 63/63 against Mistral 61/64, recovered 60/60 required evidence on both, and selected the current document 1/1 where Mistral scored 0/1. Qwen was faster (median 12449.6 ms vs 13847.6 ms) with fewer tokens. This is evidence about the transferred Mistral prompt on Qwen, not about a Qwen-adapted summarization prompt.

**Triage.** Qwen `triage.v1` transfer scored escalation 12/12 and missed escalation 0/12 against Mistral 10/12 and 2/12. Queue accuracy was 11/12 vs 10/12. Human-boundary compliance was 12/12 on both. Qwen was faster (median 4257.0 ms vs 5291.3 ms) with fewer tokens. Day 4 already rejected `triage.v2` on Mistral for extra latency with no routing gain; this harness did not re-test `triage.v2` × Qwen.

## Human boundary

`draft_reply` was scored on both Mistral and Qwen. Human-boundary compliance is 12/12 for each model. No committed reply in this run promised a refund, approved or denied a claim, said the issue was resolved, or implied a final customer outcome.
