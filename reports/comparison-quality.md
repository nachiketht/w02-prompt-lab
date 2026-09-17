## Quality

This reading is for run `local-comparison-01` with Qwen thinking off. It is not a universal ranking.

**Extraction.** Both models returned 12/12 valid objects and 0/12 repairs. Qwen `extract.v3` recovered required evidence 72/72 against Mistral `extract.v2` at 71/72, matched citations 73/73 on both, and selected the current document 1/1 where Mistral scored 0/1. Qwen also used fewer tokens per case and a lower median case latency (12218.7 ms vs 18187.5 ms). That Qwen row is an adapted prompt, not `extract.v2` transfer.

**Summarization.** Both models returned 12/12 valid objects. Qwen `summarize.v1` transfer matched citations 63/63 against Mistral 61/64, recovered 60/60 required evidence on both, and selected the current document 1/1 where Mistral scored 0/1. Qwen was faster (median 12449.6 ms vs 13847.6 ms) with fewer tokens. This is evidence about the transferred Mistral prompt on Qwen, not about a Qwen-adapted summarization prompt.

**Triage.** Qwen `triage.v1` transfer scored escalation 12/12 and missed escalation 0/12 against Mistral 10/12 and 2/12. Queue accuracy was 11/12 vs 10/12. Human-boundary compliance was 12/12 on both. Qwen was faster (median 4257.0 ms vs 5291.3 ms) with fewer tokens. Day 4 already rejected `triage.v2` on Mistral for extra latency with no routing gain; this harness did not re-test `triage.v2` × Qwen.

## Human boundary

`draft_reply` was scored on both Mistral and Qwen. Human-boundary compliance is 12/12 for each model. No committed reply in this run promised a refund, approved or denied a claim, said the issue was resolved, or implied a final customer outcome.
