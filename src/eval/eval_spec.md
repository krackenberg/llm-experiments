# Eval: RF QA/QC Pass Group

Metric goals:
- Status correctness vs. labeled gold set >= 0.9
- JSON schema validity: 100%
- For FAIL cases:
  - At least 1 reason containing a known failure keyword

Hard checks:
- status must be in {"OK","SUSPECT","FAIL"}
- JSON must parse
- For each test case, compare status to expected_status

LLM-as-judge:
- For explanation quality, a judge model rates (1–5) for:
  - specificity
  - technical correctness
  - absence of hallucinated sensor behavior
