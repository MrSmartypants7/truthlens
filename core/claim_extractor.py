"""
Claim Extractor — breaks LLM responses into individual verifiable claims.

CHANGED FROM OPENAI: Uses Ollama's local chat endpoint instead of OpenAI.
Same prompt engineering, same JSON parsing logic, just a different client.
"""

import json
import uuid

import ollama

from app.config import settings
from app.logging_config import get_logger
from models.schemas import Claim

logger = get_logger("claim_extractor")

# The extraction prompt — carefully engineered to produce structured output
EXTRACTION_SYSTEM_PROMPT = """You are a factual claim extractor. Break down text into individual verifiable factual claims.

Rules:
1. Extract ONLY factual claims (things that can be verified as true or false)
2. Skip opinions, questions, hedged statements
3. Each claim should be self-contained
4. Preserve specific numbers, dates, names exactly as stated
5. One claim per item

You MUST respond with ONLY a valid JSON object with a "claims" key containing an array.

Example input: "The Eiffel Tower, built in 1889, stands at 330 meters in Paris, France."
Example response:
{"claims": [{"claim": "The Eiffel Tower was built in 1889", "original_sentence": "The Eiffel Tower, built in 1889, stands at 330 meters in Paris, France."}, {"claim": "The Eiffel Tower is 330 meters tall", "original_sentence": "The Eiffel Tower, built in 1889, stands at 330 meters in Paris, France."}, {"claim": "The Eiffel Tower is located in Paris, France", "original_sentence": "The Eiffel Tower, built in 1889, stands at 330 meters in Paris, France."}]}"""


class ClaimExtractor:
    """Extracts verifiable claims from LLM-generated text."""

    def __init__(self):
        self.client = ollama.Client(host=settings.ollama_base_url)
        self.model = settings.llm_model

    def extract(self, text: str) -> list[Claim]:
        """
        Extract all verifiable claims from the given text.

        Pipeline:
            1. Send text to Ollama LLM with extraction prompt
            2. Parse JSON response
            3. Wrap each claim in a Claim object with unique ID
            4. Filter out any malformed results
        """
        if not text or not text.strip():
            return []

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract factual claims from this text:\n\n{text}"},
                ],
                options={"temperature": settings.llm_temperature},
                format="json",
            )

            raw_content = response["message"]["content"].strip()

            # Parse JSON — handle potential markdown code blocks
            if raw_content.startswith("```"):
                raw_content = raw_content.split("```")[1]
                if raw_content.startswith("json"):
                    raw_content = raw_content[4:]

            claims_data = json.loads(raw_content)

            # Handle case where LLM wraps array in an object
            # Local LLMs might use different key names
            if isinstance(claims_data, dict):
                # Try common key names
                for key in ["claims", "factual_claims", "extracted_claims", "results", "data"]:
                    if key in claims_data:
                        claims_data = claims_data[key]
                        break
                else:
                    # If it's a single claim wrapped in a dict
                    if "claim" in claims_data:
                        claims_data = [claims_data]
                    else:
                        # Take the first list value we find
                        for v in claims_data.values():
                            if isinstance(v, list):
                                claims_data = v
                                break
                        else:
                            claims_data = []

            logger.debug("parsed_claims_raw", count=len(claims_data) if isinstance(claims_data, list) else 0)

            claims = []
            for item in claims_data:
                if isinstance(item, dict) and "claim" in item:
                    claim = Claim(
                        claim_id=str(uuid.uuid4())[:8],
                        text=item["claim"],
                        original_sentence=item.get("original_sentence", ""),
                    )
                    claims.append(claim)

            logger.info(
                "claims_extracted",
                input_length=len(text),
                num_claims=len(claims),
            )
            return claims

        except json.JSONDecodeError as e:
            logger.error("claim_extraction_json_error", error=str(e))
            return self._fallback_extraction(text)
        except Exception as e:
            logger.error("claim_extraction_error", error=str(e))
            return self._fallback_extraction(text)

    def _fallback_extraction(self, text: str) -> list[Claim]:
        """
        Simple rule-based fallback if the LLM extraction fails.

        Treats each sentence as a claim. Less precise but better than nothing.
        """
        import re

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 10:
                claims.append(Claim(
                    claim_id=f"fallback-{i}",
                    text=sentence,
                    original_sentence=sentence,
                ))

        logger.warning(
            "using_fallback_extraction",
            num_claims=len(claims),
        )
        return claims
