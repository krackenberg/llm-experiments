"""Deterministic mock client for local / CI runs without API keys."""

import json
import hashlib
from dataclasses import dataclass


@dataclass
class MockConfig:
    model_id: str = "mock-sonnet"


# ── Canned triage responses keyed by simple content heuristics ───────────────

_TRIAGE_RESPONSES = {
    "account_access": {
        "category": "account_access",
        "priority": "P1_critical",
        "suggested_team": "security",
        "reasoning": "Enterprise customer with team-wide login failure after password reset indicates a critical authentication issue requiring immediate security team investigation.",
    },
    "billing": {
        "category": "billing",
        "priority": "P3_medium",
        "suggested_team": "billing_ops",
        "reasoning": "Duplicate charge reported by pro-tier customer. Requires billing investigation and likely refund, but not service-impacting.",
    },
    "bug_critical": {
        "category": "bug",
        "priority": "P1_critical",
        "suggested_team": "engineering",
        "reasoning": "Complete data loss in production workspace affecting 200+ enterprise users is a critical incident requiring immediate engineering response.",
    },
    "bug_high": {
        "category": "bug",
        "priority": "P2_high",
        "suggested_team": "engineering",
        "reasoning": "Dashboard rendering failure blocking daily standups for a pro-tier customer. Functional degradation requiring prompt engineering attention.",
    },
    "feature_request": {
        "category": "feature_request",
        "priority": "P4_low",
        "suggested_team": "product",
        "reasoning": "Customer requesting new export functionality. Route to product for backlog consideration.",
    },
    "general_inquiry": {
        "category": "general_inquiry",
        "priority": "P4_low",
        "suggested_team": "support_general",
        "reasoning": "Free-tier user asking about existing CSV export feature. Standard support inquiry, likely needs documentation pointer.",
    },
}

_JUDGE_RESPONSE = {"specificity": 4, "correctness": 4, "completeness": 4}

_PERSPECTIVE_RESPONSES = {
    "customer_success": {
        "sentiment": "frustrated",
        "churn_risk": "high",
        "needs_outreach": True,
        "cs_notes": "Enterprise customer experiencing critical issue. High churn risk due to team-wide impact. Proactive exec outreach recommended.",
    },
    "engineering_lead": {
        "likely_cause": "Authentication service regression after password-reset flow update",
        "blast_radius": "subset",
        "possible_regression": True,
        "fix_complexity": "moderate",
        "eng_notes": "Likely auth-service issue. Check recent deploys to password-reset or session-management services.",
    },
    "security_analyst": {
        "security_relevant": True,
        "data_exposure_risk": "low",
        "account_compromise_risk": "medium",
        "security_notes": "Team-wide lockout could indicate credential-stuffing or auth misconfiguration. Verify no unauthorized access during lockout window.",
    },
}

_SYNTHESIS_RESPONSE = {
    "common_findings": [
        "Critical issue affecting enterprise customer",
        "Authentication/access failure with team-wide blast radius",
        "High urgency requiring immediate response",
    ],
    "disagreements": [
        "Security sees possible compromise risk; engineering sees likely regression",
    ],
    "unified_priority": "P1_critical",
    "action_plan": [
        "Immediately engage security team to rule out compromise",
        "Engineering to check recent auth-service deploys for regressions",
        "Customer Success to send proactive update to customer exec sponsor",
        "Set up bridge call if not resolved within 30 minutes",
    ],
    "escalation_needed": True,
}


def _extract_ticket(text: str) -> str:
    """Pull the ticket body out of the full prompt, if present."""
    # Look for <TICKET>...</TICKET> wrapper from triage prompt
    import re
    m = re.search(r"<TICKET>\s*(.*?)\s*</TICKET>", text, re.DOTALL)
    if m:
        return m.group(1)
    # Look for TICKET: or ORIGINAL TICKET: section
    m = re.search(r"(?:ORIGINAL )?TICKET:\s*\n(.*?)(?:\n\n|$)", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def _classify_ticket(text: str) -> str:
    """Simple keyword classifier to pick a canned response."""
    t = _extract_ticket(text).lower()
    if "password" in t or "log in" in t or "locked out" in t or "credentials" in t:
        return "account_access"
    if "charged twice" in t or "refund" in t or "billing" in t or "charge" in t or "payment" in t:
        return "billing"
    if ("data missing" in t or "workspace appears empty" in t or "all data" in t
            or "outage" in t or "503" in t or "everything is down" in t or "total outage" in t):
        return "bug_critical"
    if "not loading" in t or "spinner" in t or "dashboard" in t:
        return "bug_high"
    if "feature" in t or "would be nice" in t:
        return "feature_request"
    if ("how do i" in t or "export" in t or "csv" in t or "new to" in t
            or "where" in t or "setting" in t or "profile" in t or "avatar" in t):
        return "general_inquiry"
    # Fallback: deterministic hash-based pick
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    keys = sorted(_TRIAGE_RESPONSES)
    return keys[h % len(keys)]


def _is_judge_prompt(text: str) -> bool:
    return "rate the triage" in text.lower() or "specificity" in text.lower()


def _is_perspective_prompt(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("customer success manager", "engineering lead", "security analyst"))


def _get_perspective_id(text: str) -> str:
    t = text.lower()
    if "customer success" in t:
        return "customer_success"
    if "engineering lead" in t:
        return "engineering_lead"
    return "security_analyst"


def _is_synthesis_prompt(text: str) -> bool:
    return "synthesiz" in text.lower() or "unified assessment" in text.lower()


class MockClient:
    """Drop-in replacement for AnthropicClient that returns deterministic responses."""

    def __init__(self, cfg: MockConfig | None = None):
        self.cfg = cfg or MockConfig()

    def chat(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        user_text = messages[-1]["content"] if messages else ""

        if _is_judge_prompt(user_text):
            return json.dumps(_JUDGE_RESPONSE)

        if _is_synthesis_prompt(user_text):
            return json.dumps(_SYNTHESIS_RESPONSE)

        if _is_perspective_prompt(user_text):
            pid = _get_perspective_id(user_text)
            return json.dumps(_PERSPECTIVE_RESPONSES.get(pid, _PERSPECTIVE_RESPONSES["security_analyst"]))

        # Default: triage
        key = _classify_ticket(user_text)
        return json.dumps(_TRIAGE_RESPONSES[key])
