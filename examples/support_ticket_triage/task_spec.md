# Task: Support Ticket Triage

**Business goal:** Automatically classify incoming support tickets so they route
to the correct team and get the right priority, reducing mean-time-to-response.

## Input
A support ticket containing:
- `subject`: one-line summary
- `body`: customer's description of the issue
- `customer_tier`: "free" | "pro" | "enterprise"

## Output
JSON with:
- `category`: one of "billing", "bug", "feature_request", "account_access", "general_inquiry"
- `priority`: "P1_critical" | "P2_high" | "P3_medium" | "P4_low"
- `reasoning`: 1–2 sentence explanation
- `suggested_team`: "billing_ops", "engineering", "product", "security", "support_general"

## Priority rules
- P1: service outage, data loss, security breach, or enterprise customer blocked
- P2: significant functionality broken, or pro/enterprise customer impacted
- P3: minor bug, feature request from paying customer
- P4: general inquiry, feature request from free tier
