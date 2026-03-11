"""
Verification Engine — the heart of TruthLens.

CHANGED FROM OPENAI: Uses Ollama's local chat endpoint.
Same verification logic, same prompt engineering, just free and local.
"""

import json

import ollama

from app.config import settings
from app.logging_config import get_logger
from models.schemas import (
    Claim,
    ClaimVerification,
    RetrievedEvidence,
    SeverityLevel,
    VerificationStatus,
)

logger = get_logger("verifier")

VERIFICATION_SYSTEM_PROMPT = """You are a factual claim verifier. You receive a CLAIM and EVIDENCE documents.

Determine if the evidence SUPPORTS, CONTRADICTS, or is INSUFFICIENT to verify the claim.

Rules:
- Only use the provided evidence. Do not use your own knowledge.
- Be precise about numbers, dates, and names.
- If the evidence is close but not exact (e.g., claim says "324m", evidence says "330m"), mark as "contradicted".
- If no evidence is relevant, mark as "unverifiable".

You MUST respond with ONLY a valid JSON object. No other text. Every string value must be in double quotes.

Example of correct response format:
{"status": "contradicted", "confidence": 0.9, "severity": "medium", "reasoning": "Claim says 324m but evidence says 330m"}

Valid status values: "supported", "contradicted", "partially_supported", "unverifiable"
Valid severity values: "low", "medium", "high", "critical"
confidence must be a number between 0.0 and 1.0"""


class VerificationEngine:
    """Verifies individual claims against retrieved evidence."""

    def __init__(self):
        self.client = ollama.Client(host=settings.ollama_base_url)
        self.model = settings.llm_model

    def verify_claim(
        self,
        claim: Claim,
        evidence: list[RetrievedEvidence],
    ) -> ClaimVerification:
        """
        Verify a single claim against retrieved evidence.
        """
        # If no evidence was found, the claim is unverifiable
        if not evidence:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.5,
                severity=SeverityLevel.MEDIUM,
                evidence=[],
                reasoning="No relevant evidence found in the knowledge base.",
            )

        # Format evidence for the prompt
        evidence_text = self._format_evidence(evidence)

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"CLAIM: {claim.text}\n\n"
                            f"EVIDENCE:\n{evidence_text}\n\n"
                            f"Verify this claim against the evidence. Respond with ONLY valid JSON."
                        ),
                    },
                ],
                options={"temperature": settings.llm_temperature},
                format="json",
            )

            raw = response["message"]["content"].strip()
            result = self._parse_verification_response(raw)

            verification = ClaimVerification(
                claim=claim,
                status=VerificationStatus(result["status"]),
                confidence=float(result["confidence"]),
                severity=SeverityLevel(result.get("severity", "low")),
                evidence=evidence,
                reasoning=result.get("reasoning", ""),
            )

            logger.info(
                "claim_verified",
                claim_id=claim.claim_id,
                status=verification.status.value,
                confidence=verification.confidence,
            )
            return verification

        except Exception as e:
            logger.error(
                "verification_error",
                claim_id=claim.claim_id,
                error=str(e),
            )
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                severity=SeverityLevel.MEDIUM,
                evidence=evidence,
                reasoning=f"Verification failed: {str(e)}",
            )

    def _format_evidence(self, evidence: list[RetrievedEvidence]) -> str:
        """Format evidence chunks into a readable string for the prompt."""
        parts = []
        for e in evidence:
            parts.append(
                f"[Source: {e.chunk.source} | Relevance rank: {e.rank}]\n"
                f"{e.chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _parse_verification_response(self, raw: str) -> dict:
        """
        Parse the LLM's verification response into a structured dict.

        Handles common issues with local LLMs (Ollama/llama3.2):
          - Markdown code blocks around JSON
          - Unquoted string values (e.g., "reasoning": The claim says...)
          - Extra text before/after the JSON
          - Truncated responses
        """
        import re

        # Strip markdown code blocks
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Extract JSON object from surrounding text
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = raw[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            # ── Fix unquoted string values ────────────────────────────────
            # Local LLMs often produce: "reasoning": The claim says 324m...
            # instead of:               "reasoning": "The claim says 324m..."
            # We fix this by finding unquoted values and wrapping them.
            try:
                fixed = re.sub(
                    r'("(?:reasoning|status|severity)":\s*)([^"{\[\d][^,}]*?)(\s*[,}])',
                    lambda m: m.group(1) + '"' + m.group(2).strip().replace('"', '\\"') + '"' + m.group(3),
                    json_str,
                )
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        # ── Regex extraction as last resort ───────────────────────────────
        # If JSON is totally broken, pull out individual fields with regex
        result = {}

        status_match = re.search(
            r'"status"\s*:\s*"?(supported|contradicted|partially_supported|unverifiable)"?',
            raw, re.IGNORECASE,
        )
        if status_match:
            result["status"] = status_match.group(1).lower()

        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
        if conf_match:
            result["confidence"] = float(conf_match.group(1))

        severity_match = re.search(
            r'"severity"\s*:\s*"?(low|medium|high|critical)"?',
            raw, re.IGNORECASE,
        )
        if severity_match:
            result["severity"] = severity_match.group(1).lower()

        reasoning_match = re.search(
            r'"reasoning"\s*:\s*"?(.+?)(?:"\s*[,}]|$)',
            raw, re.DOTALL,
        )
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip().strip('"')

        if "status" in result:
            # We got at least the status — fill in defaults for missing fields
            result.setdefault("confidence", 0.7)
            result.setdefault("severity", "medium")
            result.setdefault("reasoning", "Parsed from malformed response.")
            logger.debug("parsed_with_regex", status=result["status"])
            return result

        # Complete failure
        logger.warning("failed_to_parse_verification", raw_response=raw[:200])
        return {
            "status": "unverifiable",
            "confidence": 0.0,
            "severity": "medium",
            "reasoning": "Failed to parse verification response.",
        }

    def compute_overall_score(
        self, verifications: list[ClaimVerification]
    ) -> float:
        """
        Compute an overall reliability score for the entire LLM response.

        Score range: 0.0 (all contradicted) to 1.0 (all supported)
        """
        if not verifications:
            return 0.5

        total_weight = 0.0
        weighted_score = 0.0

        for v in verifications:
            weight = v.confidence

            if v.status == VerificationStatus.SUPPORTED:
                weighted_score += weight * 1.0
                total_weight += weight
            elif v.status == VerificationStatus.CONTRADICTED:
                weighted_score += weight * 0.0
                total_weight += weight
            elif v.status == VerificationStatus.PARTIALLY_SUPPORTED:
                weighted_score += weight * 0.5
                total_weight += weight
            else:  # UNVERIFIABLE
                weighted_score += weight * 0.3
                total_weight += weight

        if total_weight == 0:
            return 0.5

        return round(weighted_score / total_weight, 3)
