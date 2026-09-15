# Q3 Full failed-step refill

Do not delete successful cached steps or change the evaluation denominator.

## Inputs
- checkpoint: `${RECREDIT_ROOT}/ckpts/.../full/best`
- failed-step list: `results/rh20t_metrics/q3_full_FAILED_STEPS.json`
- refill only steps with `request_error != null`; keep successful caches
- store full `response`, HTTP status, and `error`
- merge back into `q3_full_ol/<id>/result.json`, then re-score with the same v1.1 parser

## Protocol
When GPUs are free, use the existing `run_rh20t_*` offline protocol and refill
failed steps only (do not contend with concurrent n781 / TYPE-NEG jobs).
