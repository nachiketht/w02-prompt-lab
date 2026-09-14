# Day 2 comparison: Mistral vs Qwen

During the initial run, with max_output_tokens set at 256, Mistral was able to complete all its outputs whereas Qwen's output got truncated in all 12 responses because thinking is on by default on Qwen.

This value was then updated to 512 and a few responses of Qwen were still getting truncated. I then proceeded to increase it to 1024 and all the responses passed with a comfortble headroom of around 300.

I would've considered increasing it to 2048 if there were still truncated outputs in the 1024 case, but since there was a headroom of ~300, i decided 2048 would be overkill.

I believe, for a baseline to be accurate, the model needs to be able to generate its output. If there is no output, the baseline does not make sense. And, on the thinking perspective, Qwen is by default on think mode. If we are comparing two models on their baseline performance, we must use the default value. Just because Mistral doesn't have thinking, doesnt mean we have to switch it off for Qwen. The test is not about fairness, its about a performance baseline.

## Workload


| Model        | Cases succeeded | Input tokens (total / min / max) | Output tokens (total / min / max) | Latency ms (median / max) |
| ------------ | --------------- | -------------------------------- | --------------------------------- | ------------------------- |
| `mistral:7b` | 12 / 12         | 2775 / 200 / 274                 | 1318 / 76 / 188                   | 5394 / 10474              |
| `qwen3:8b`   | 12 / 12         | 2487 / 178 / 244                 | 5633 / 282 / 733                  | 26760 / 73795             |


Qwen used about 4× as many output tokens and about 6× as much wall time as Mistral on the same cases. Mistral’s longest answer was 188 tokens (S01). Qwen’s shortest was 282 (S05) and its longest was 733 (S01). That gap is mostly Qwen3 thinking tokens counted in `eval_count`; the visible `response_text` is a short field list on both models. Changing models did not require changing the caller-only `model_id`.

## Answers

For this baseline, Mistral is the cheaper local completer; Qwen is complete at 1024 tokens but much slower because thinking consumes the output budget before the short answer appears.