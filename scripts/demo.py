"""
TruthLens Demo Script — shows the full pipeline in action using Ollama.

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull models:
         ollama pull llama3.2
         ollama pull nomic-embed-text
    3. Run this script:
         cd truthlens
         python scripts/demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pipeline import TruthLensPipeline
from models.schemas import VerifyRequest


# ─── Sample Knowledge Base ────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    {
        "text": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
            "in Paris, France. It is named after the engineer Gustave Eiffel, whose "
            "company designed and built the tower from 1887 to 1889 as the centerpiece "
            "of the 1889 World's Fair. The tower is 330 metres (1,083 ft) tall, about "
            "the same height as an 81-storey building. It was the first structure in "
            "the world to reach a height of 300 metres."
        ),
        "source": "wikipedia",
        "metadata": {"title": "Eiffel Tower", "topic": "landmarks"},
    },
    {
        "text": (
            "The Great Wall of China is a series of fortifications that were built "
            "across the historical northern borders of ancient Chinese states and "
            "Imperial China as protection against various nomadic groups. The total "
            "length of all sections ever built is over 21,196 kilometres (13,171 mi). "
            "The best-known sections were built by the Ming dynasty (1368-1644). "
            "Contrary to popular myth, the Great Wall is not visible from space with "
            "the naked eye."
        ),
        "source": "wikipedia",
        "metadata": {"title": "Great Wall of China", "topic": "landmarks"},
    },
    {
        "text": (
            "Python is a high-level, general-purpose programming language. Its design "
            "philosophy emphasizes code readability with the use of significant "
            "indentation. Python is dynamically typed and garbage-collected. It supports "
            "multiple programming paradigms, including structured, object-oriented, and "
            "functional programming. Guido van Rossum began working on Python in the "
            "late 1980s as a successor to the ABC programming language and first "
            "released it in 1991 as Python 0.9.0."
        ),
        "source": "wikipedia",
        "metadata": {"title": "Python (programming language)", "topic": "technology"},
    },
    {
        "text": (
            "The human heart beats approximately 100,000 times per day, pumping about "
            "2,000 gallons of blood through approximately 60,000 miles of blood vessels. "
            "The average adult heart weighs between 8 and 12 ounces (230-340 grams). "
            "The heart has four chambers: two atria (upper chambers) and two ventricles "
            "(lower chambers). The left ventricle is the strongest chamber."
        ),
        "source": "medical-reference",
        "metadata": {"title": "Human Heart", "topic": "biology"},
    },
]


# ─── LLM Responses to Verify (mix of correct and incorrect) ──────────────────

TEST_RESPONSES = [
    {
        "query": "Tell me about the Eiffel Tower",
        "llm_response": (
            "The Eiffel Tower is located in Paris, France. It stands at 324 meters "
            "tall and was completed in 1889. The tower was designed by Gustave Eiffel "
            "for the 1889 World's Fair. It is made of wrought iron."
        ),
        "notes": "324m is wrong (should be 330m), rest is correct",
    },
    {
        "query": "Tell me about the Great Wall of China",
        "llm_response": (
            "The Great Wall of China is over 13,000 miles long and was primarily "
            "built during the Qin dynasty. It is clearly visible from space with "
            "the naked eye. The wall was built to protect against nomadic invasions."
        ),
        "notes": "Contains the famous myth about being visible from space",
    },
]


def run_demo():
    """Run the full TruthLens demo."""
    print("=" * 70)
    print("  TruthLens — Hallucination Detection Demo (Ollama)")
    print("=" * 70)

    # Check if Ollama is running
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        client.list()
        print("\n✅ Ollama is running")
    except Exception as e:
        print(f"\n❌ Cannot connect to Ollama: {e}")
        print("\nMake sure Ollama is installed and running:")
        print("  1. Download from https://ollama.com")
        print("  2. Run: ollama pull llama3.2")
        print("  3. Run: ollama pull nomic-embed-text")
        print("  4. Then run this script again")
        return

    # Initialize pipeline
    print("\n📦 Initializing TruthLens pipeline...")
    pipeline = TruthLensPipeline()

    # Ingest documents
    print(f"\n📚 Ingesting {len(SAMPLE_DOCUMENTS)} documents...")
    result = pipeline.ingest_documents(SAMPLE_DOCUMENTS)
    print(f"   ✅ Created {result.chunks_created} chunks")
    print(f"   ✅ Index size: {result.index_size} vectors")

    # Verify each test response
    for i, test in enumerate(TEST_RESPONSES):
        print(f"\n{'─' * 70}")
        print(f"  Test {i+1}: {test['query']}")
        print(f"  Note: {test['notes']}")
        print(f"{'─' * 70}")
        print(f"\n  LLM Response: {test['llm_response'][:100]}...")

        request = VerifyRequest(
            llm_response=test["llm_response"],
            query=test["query"],
        )
        response = pipeline.verify(request)

        print(f"\n  📊 Results:")
        print(f"     Reliability Score: {response.overall_reliability_score:.1%}")
        print(f"     Total Claims: {response.total_claims}")
        print(f"     ✅ Supported: {response.supported_claims}")
        print(f"     ❌ Contradicted: {response.contradicted_claims}")
        print(f"     ❓ Unverifiable: {response.unverifiable_claims}")
        print(f"     ⏱️  Time: {response.verification_time_ms:.0f}ms")

        for v in response.claim_verifications:
            status_icon = {
                "supported": "✅",
                "contradicted": "❌",
                "unverifiable": "❓",
                "partially_supported": "⚠️",
            }.get(v.status.value, "?")

            print(f"\n     {status_icon} \"{v.claim.text}\"")
            print(f"        Status: {v.status.value} (confidence: {v.confidence:.0%})")
            if v.reasoning:
                print(f"        Reason: {v.reasoning[:120]}")

    print(f"\n{'=' * 70}")
    print("  Demo complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_demo()
