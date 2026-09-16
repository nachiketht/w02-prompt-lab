# Day 4 notes

Run `a35fb087-e0ff-4b1f-a5a9-0a2d73ced15d` used `mistral:7b` at temperature `0.0` for both prompt versions. `max_output_tokens` was `1024`. Provider/API cost: `$0.00`.

triage.v1
queue correct: 10/12
escalation correct: 10/12
missed escalations: 2
unnecessary escalations: 0
human-boundary passes: 12/12

triage.v2
queue correct: 10/12
escalation correct: 10/12
missed escalations: 2
unnecessary escalations: 0
human-boundary passes: 12/12

changed-queue count: 0
output tokens: v1 1554 total, v2 2165 total, difference (v2-v1) 611
median latency: v1 5791 ms, v2 14826.5 ms
maximum latency: v1 8311 ms, v2 21406 ms
observation count: 12 per version (24 CallRecords, no transport retries or schema repairs)

Both versions missed T06 and T07, which gold labels as `escalate`. v2's analysis field noticed mixed intent on those cases and still chose a single product queue (`card_dispute` and `complaint`). Routing quality did not change. v2 spent 611 extra output tokens and about 2.5 times the median latency. On this 12-case set, the additional analysis field did not earn that overhead. A zero-case routing difference is not evidence that either prompt is universally better.
