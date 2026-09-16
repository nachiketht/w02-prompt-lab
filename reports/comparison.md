# Model Comparison

/-----------------------------------
## Observations:
I used same prompt on extraction for both. Retried were missing from the output. Truncation errors occured in qwen because the prompt was not adopted. 
/-----------------------------------

Run ID: `local-comparison-01`

Counts are reported with their denominators. Headline latency is median and maximum **case** end-to-end time (`elapsed_ms`), with one observation per case. Attempt latency is HTTP-call time and uses a separate `n`. Mean latency is not used.

Provider/API charge is `$0.00`.

## Extraction

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median case latency | Max case latency | Case n | Median attempt latency | Max attempt latency | Attempt n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | extract.v2 | 12/12 | citation_correctness: 73/73<br>document_status_accuracy: 9/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 71/72<br>unsupported_field_avoidance: 10/12<br>version_selection_accuracy: 0/1 | 2292.9 | 355.9 | 18453.7 ms | 19860.6 ms | 12 | 18449.5 ms | 19857 ms | 12 | 0/12 | 0 | 0 |
| qwen | extract.v2 transfer | 6/12 | citation_correctness: 34/34<br>document_status_accuracy: 5/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 33/72<br>unsupported_field_avoidance: 8/12<br>version_selection_accuracy: 1/1 | 1918.9 | 907.9 | 47139.2 ms | 49902.5 ms | 12 | 47135.5 ms | 49899 ms | 12 | 0/12 | 0 | 6 |

## Summarization

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median case latency | Max case latency | Case n | Median attempt latency | Max attempt latency | Attempt n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | summarize.v1 | 12/12 | citation_correctness: 61/64<br>document_status_accuracy: 8/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 60/60<br>unsupported_field_avoidance: 8/12<br>version_selection_accuracy: 0/1 | 1172.2 | 298.2 | 14053.5 ms | 17704.4 ms | 12 | 14051 ms | 17700 ms | 12 | 0/12 | 0 | 0 |
| qwen | summarize.v1 transfer | 11/12 | citation_correctness: 56/56<br>document_status_accuracy: 9/12<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 54/60<br>unsupported_field_avoidance: 10/12<br>version_selection_accuracy: 1/1 | 1011.2 | 638.2 | 25187.5 ms | 46675.1 ms | 12 | 25186 ms | 46672 ms | 12 | 0/12 | 0 | 1 |

## Triage

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median case latency | Max case latency | Case n | Median attempt latency | Max attempt latency | Attempt n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | triage.v1 | 12/12 | escalation_accuracy: 10/12<br>human_boundary_compliance: 12/12<br>missed_escalation: 2/12 ↓<br>pii_leakage: 0/12 ↓<br>queue_accuracy: 10/12<br>unnecessary_escalation: 0/12 ↓ | 789.8 | 129.5 | 5287.9 ms | 6873.6 ms | 12 | 5286.5 ms | 6871 ms | 12 | 0/12 | 0 | 0 |
| qwen | triage.v1 transfer | 11/12 | escalation_accuracy: 11/12<br>human_boundary_compliance: 11/12<br>missed_escalation: 0/12 ↓<br>pii_leakage: 0/12 ↓<br>queue_accuracy: 11/12<br>unnecessary_escalation: 0/12 ↓ | 692.6 | 485.5 | 17377.0 ms | 43824.7 ms | 12 | 17374.5 ms | 43816 ms | 12 | 0/12 | 0 | 1 |

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
