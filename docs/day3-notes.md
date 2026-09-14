# Day 3 notes

Run `33cdbb63-e697-4aae-a569-c42bc2205232` used `mistral:7b` at temperature `0.0`.

- Summarization repair rate: **0/12**
- Extraction repair rate: **0/12**
- Example leakage count: **0**
- Citation-existence failure count: **3** (all on summarization; extraction had 0)

The most common validation error, seen when the prompt received a raw JSON Schema dump, was extra forbidden keys such as `$defs`, `properties`, and `type` because the model echoed the schema instead of an instance. Generating an instance-oriented field listing from the Pydantic models, and telling the model not to return JSON Schema keys, removed that failure mode so the scored run needed no semantic repairs.
