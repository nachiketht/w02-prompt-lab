# Model Decision Record

Run ID: `local-comparison-01`

Do not select one universal model solely because it leads on a different task. Each decision below is for one task, one prompt version, and this 12-case local run with Qwen `think=False`.

## Evaluated models

- mistral (`mistral:7b`, think omitted)
- qwen (`qwen3:8b`, think=False)

## Evaluated configurations

- `extraction` — mistral — `extract.v2`
- `extraction` — qwen — `extract.v3`
- `summarization` — mistral — `summarize.v1`
- `summarization` — qwen — `summarize.v1 transfer`
- `triage` — mistral — `triage.v1`
- `triage` — qwen — `triage.v1 transfer`

## Evidence

From `reports/comparison.md` for `local-comparison-01`. Provider/API charge `$0.00`. Case latency n=12. No repairs or transport retries. Human-boundary compliance 12/12 on both models.

- Extraction / mistral / `extract.v2`: valid 12/12, recall 71/72, citations 73/73, unsupported avoidance 10/12, version 0/1, median case 18187.5 ms
- Extraction / qwen / `extract.v3`: valid 12/12, recall 72/72, citations 73/73, unsupported avoidance 11/12, version 1/1, median case 12218.7 ms
- Summarization / mistral / `summarize.v1`: valid 12/12, recall 60/60, citations 61/64, version 0/1, median case 13847.6 ms
- Summarization / qwen / `summarize.v1 transfer`: valid 12/12, recall 60/60, citations 63/63, version 1/1, median case 12449.6 ms
- Triage / mistral / `triage.v1`: queue 10/12, escalation 10/12, missed 2/12, unnecessary 0/12, human boundary 12/12, median case 5291.3 ms
- Triage / qwen / `triage.v1 transfer`: queue 11/12, escalation 12/12, missed 0/12, unnecessary 0/12, human boundary 12/12, median case 4257.0 ms

Day 4 constraint kept: `triage.v2` did not earn its latency on Mistral (same 10/12 routing, more tokens, ~2.5× median latency). That comparison is not reopened here.

## Task decisions

### Extraction

- selected model: qwen
- prompt version: `extract.v3`
- measured reason: 12/12 valid JSON, required-evidence recall 72/72 vs 71/72, version selection 1/1 vs 0/1, lower median and max case latency, fewer tokens per case, 0/12 repairs
- rejected alternative(s): mistral `extract.v2` (home prompt; slightly lower recall and failed version selection on this set)
- condition that would reopen the decision: Qwen `extract.v2` transfer under think-off, a think-on rerun, or a larger case set where `extract.v3` schema fit no longer holds

### Summarization

- selected model: qwen
- prompt version: `summarize.v1` transfer
- measured reason: 12/12 valid JSON, citation 63/63 vs 61/64, version selection 1/1 vs 0/1, lower median case latency, fewer tokens, 0/12 repairs
- rejected alternative(s): mistral `summarize.v1`. This is not a claim that Qwen has a better summarization prompt; the Qwen row used the transferred Mistral version
- condition that would reopen the decision: a Qwen-adapted summarization prompt, think-on, or a set large enough that 61/64 vs 63/63 is not treated as directional

### Triage

- selected model: qwen
- prompt version: `triage.v1` transfer
- measured reason: escalation 12/12 vs 10/12, missed escalation 0/12 vs 2/12, queue 11/12 vs 10/12, human boundary 12/12 on both, lower median case latency, fewer tokens
- rejected alternative(s): mistral `triage.v1`. `triage.v2` stays rejected from Day 4 and was not run on Qwen
- condition that would reopen the decision: `triage.v2` × Qwen (untested), a gold set where T06/T07 mix-intent cases change, or production-volume latency needs
