# Task: RF Collection QA/QC for Pass Groups

Input:
- Spectrogram image(s)
- Metadata: collection time, platform, frequency band, expected signal type

Goal:
- Determine whether the collection likely succeeded or failed.
- Explain reasoning in 2–3 bullets.
- Be harsh: err on flagging questionable data.

Output:
- JSON with fields:
  - status: "OK" | "SUSPECT" | "FAIL"
  - reasons: [string]
  - suggested_followup: [string]
