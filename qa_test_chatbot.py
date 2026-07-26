"""
qa_test_chatbot.py
═══════════════════════════════════════════════════════
Senior QA Engineer Test Suite for Fast Sales AI Chatbot
Tests that ALL responses come STRICTLY from the Developer Manual PDF.

Test Categories:
  1. Direct Refund Questions (exact match from PDF)
  2. Rephrased/Confused Refund Questions (natural language variations)
  3. Cross-Topic Misdirection (trick questions to catch hallucination)
  4. Course Access & Duration (previously failed — pulled from info.json)
  5. Certificate & Employment Boundaries
  6. Edge Cases & Adversarial Questions
"""

from __future__ import annotations
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Ensure encoding works on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from main_agents import (
    create_conversation_state,
    process_prompt,
    normalize_user_info,
    get_openai_client,
)

# ── Test Configuration ────────────────────────────────────────────────────────
TEST_USER_INFO = normalize_user_info("QA Tester", "qa@test.com", "555-000-0000")
CLIENT = get_openai_client()

# Keywords that should NEVER appear in responses (info.json-only data)
FORBIDDEN_PHRASES = [
    "10 hours video content",
    "14 modules",
    "duration_hours",
    "total_modules",
    '"price": "249.00"',
    '"old_price": "500.00"',
    "Digital Marketing Essentials for Beginners",
    "SEO, content creation, email campaigns",
]

# ── Test Cases ────────────────────────────────────────────────────────────────
TEST_CASES = [
    # ═══════════════════════════════════════════════════════
    # CATEGORY 1: Direct Refund Questions (From PDF Q&A)
    # ═══════════════════════════════════════════════════════
    {
        "id": "REF-001",
        "category": "Direct Refund",
        "question": "Can I get a refund because I was not hired?",
        "must_contain_any": [
            "independently determined by dealerships",
            "Refund Policy",
            "employment outcomes",
        ],
        "must_not_contain": ["guaranteed", "yes you can", "we will process"],
    },
    {
        "id": "REF-002",
        "category": "Direct Refund",
        "question": "Can I get a refund because I was not promoted?",
        "must_contain_any": [
            "independently determined by employers",
            "Refund Policy",
            "promotions",
        ],
        "must_not_contain": ["yes", "we will refund"],
    },
    {
        "id": "REF-003",
        "category": "Direct Refund",
        "question": "I didn't use the course. Can I get a refund?",
        "must_contain_any": [
            "Refunds are not based on usage",
            "immediately after purchase",
            "course completion status",
        ],
        "must_not_contain": ["yes", "we will process your refund"],
    },
    {
        "id": "REF-004",
        "category": "Direct Refund",
        "question": "I forgot I purchased the course and want a refund.",
        "must_contain_any": [
            "digital educational products",
            "considered final",
            "platform policies",
        ],
        "must_not_contain": ["we will refund", "here is your refund"],
    },
    {
        "id": "REF-005",
        "category": "Direct Refund",
        "question": "I completed the Auto Sales course, but was not hired. Can I receive a refund?",
        "must_contain_any": [
            "do not guarantee interviews",
            "hiring",
            "Refunds are not provided based on hiring results",
        ],
        "must_not_contain": ["yes", "we can process"],
    },
    {
        "id": "REF-006",
        "category": "Direct Refund",
        "question": "Can dealerships request refunds if employee performance does not improve?",
        "must_contain_any": [
            "Training outcomes vary",
            "employee participation",
            "implementation",
        ],
        "must_not_contain": ["yes", "refund approved"],
    },
    {
        "id": "REF-007",
        "category": "Direct Refund",
        "question": "Are refunds available for unused subscription time?",
        "must_contain_any": [
            "Subscription billing",
            "refund eligibility",
            "platform policies",
        ],
        "must_not_contain": ["yes, we offer full refunds"],
    },
    {
        "id": "REF-008",
        "category": "Direct Refund",
        "question": "I forgot about my expiration date. Can I receive a refund?",
        "must_contain_any": [
            "Enrollment periods begin immediately",
            "not based on course usage",
            "completion status",
            "inactivity",
        ],
        "must_not_contain": ["yes", "we will extend"],
    },

    # ═══════════════════════════════════════════════════════
    # CATEGORY 2: Rephrased / Confused Refund Questions
    # ═══════════════════════════════════════════════════════
    {
        "id": "CONF-001",
        "category": "Confused Refund",
        "question": "Why won't you give me my money back? I never even used the course!",
        "must_contain_any": [
            "Refunds are not based on usage",
            "digital",
            "immediately after purchase",
        ],
        "must_not_contain": ["we will process", "here is your refund"],
    },
    {
        "id": "CONF-002",
        "category": "Confused Refund",
        "question": "I changed my mind about the course, what's your return policy?",
        "must_contain_any": [
            "digital educational products",
            "considered final",
            "online learning platform",
        ],
        "must_not_contain": ["30-day return", "money-back guarantee"],
    },
    {
        "id": "CONF-003",
        "category": "Confused Refund",
        "question": "Can I dispute the payment with my bank if you won't refund me?",
        "must_contain_any": [
            "contact Fast Sales Training Center Support first",
            "chargebacks",
            "disputes",
        ],
        "must_not_contain": ["go ahead and dispute", "we encourage you to call your bank"],
    },
    {
        "id": "CONF-004",
        "category": "Confused Refund",
        "question": "What happens if I file a chargeback against you?",
        "must_contain_any": [
            "restricted",
            "payment dispute",
            "fraud",
        ],
        "must_not_contain": ["nothing will happen", "we won't take action"],
    },
    {
        "id": "CONF-005",
        "category": "Confused Refund",
        "question": "Why can't I get a refund even though I wasn't hired after finishing your sales course?",
        "must_contain_any": [
            "educational product",
            "professional development",
            "not a guaranteed employment",
        ],
        "must_not_contain": ["we will make an exception"],
    },

    # ═══════════════════════════════════════════════════════
    # CATEGORY 3: Cross-Topic Misdirection / Trick Questions
    # ═══════════════════════════════════════════════════════
    {
        "id": "TRICK-001",
        "category": "Trick Question",
        "question": "Can Fast Sales Training Center process my Amazon refund?",
        "must_contain_any": [
            "Amazon manages all",
            "book orders",
            "shipping",
            "returns",
            "refunds directly",
        ],
        "must_not_contain": ["yes we can help", "send us your order number"],
    },
    {
        "id": "TRICK-002",
        "category": "Trick Question",
        "question": "Will this course guarantee me a sales job at a dealership?",
        "must_contain_any": [
            "educational and professional development purposes only",
            "independently determined",
            "employer",
        ],
        "must_not_contain": ["yes", "guaranteed job", "we will place you"],
    },
    {
        "id": "TRICK-003",
        "category": "Trick Question",
        "question": "Can Fast Sales Training Center contact a dealership to help me get hired?",
        "must_contain_any": [
            "employment and hiring decisions remain solely",
            "discretion of each dealership",
        ],
        "must_not_contain": ["yes we can call", "we will reach out"],
    },

    # ═══════════════════════════════════════════════════════
    # CATEGORY 4: Course Access & Duration (CRITICAL — was failing)
    # ═══════════════════════════════════════════════════════
    {
        "id": "ACCESS-001",
        "category": "Course Access",
        "question": "How long do I have access to the course?",
        "must_contain_any": [
            "90 days",
            "date of purchase",
            "dealership account remains active",
        ],
        "must_not_contain": ["10 hours", "14 modules", "Lifetime"],
    },
    {
        "id": "ACCESS-002",
        "category": "Course Access",
        "question": "Are the courses self-paced?",
        "must_contain_any": [
            "self-paced",
            "your schedule",
        ],
        "must_not_contain": ["10 hours video content"],
    },
    {
        "id": "ACCESS-003",
        "category": "Course Access",
        "question": "Do I receive a certificate after completing the course?",
        "must_contain_any": [
            "certificate",
            "successful course completion",
        ],
        "must_not_contain": [],
    },

    # ═══════════════════════════════════════════════════════
    # CATEGORY 5: Certificate & Employment Boundaries
    # ═══════════════════════════════════════════════════════
    {
        "id": "CERT-001",
        "category": "Certificate",
        "question": "Does earning a certificate guarantee employment?",
        "must_contain_any": [
            "successful course completion",
            "training participation",
            "independently determined",
        ],
        "must_not_contain": ["yes", "guaranteed employment"],
    },
    {
        "id": "CERT-002",
        "category": "Certificate",
        "question": "Are Fast Sales Training Center certificates government-issued certifications?",
        "must_contain_any": [
            "independent educational training programs",
            "provided by Fast Sales Training Center",
        ],
        "must_not_contain": ["yes", "government certified", "accredited by the state"],
    },

    # ═══════════════════════════════════════════════════════
    # CATEGORY 6: Edge Cases & Adversarial Questions
    # ═══════════════════════════════════════════════════════
    {
        "id": "EDGE-001",
        "category": "Edge Case",
        "question": "Can Fast Sales Training Center guarantee me a $100,000 salary?",
        "must_contain_any": [
            "not guarantee",
            "income",
            "sales performance",
        ],
        "must_not_contain": ["yes", "$100,000", "we guarantee"],
    },
    {
        "id": "EDGE-002",
        "category": "Edge Case",
        "question": "I want to know the exact price of each course and how many hours of video content they have",
        "must_contain_any": [
            # Should NOT pull info.json data — should reference what the manual says or admit it doesn't have that detail
        ],
        "must_not_contain": ["249.00", "500.00", "10 hours video content", "14 modules"],
    },
    {
        "id": "EDGE-003",
        "category": "Edge Case",
        "question": "Does applying through the jobs platform guarantee an interview?",
        "must_contain_any": [
            "do not guarantee interviews",
            "employment opportunities",
        ],
        "must_not_contain": ["yes", "guaranteed interview"],
    },
]


# ── Test Runner ───────────────────────────────────────────────────────────────

def run_single_test(test_case: dict) -> dict:
    """Run a single test case and return the result."""
    test_id = test_case["id"]
    question = test_case["question"]

    state = create_conversation_state(
        role="student",
        user_info=TEST_USER_INFO,
        agent_name="Sophia",
        include_confirmation=True,
    )

    try:
        response = process_prompt(
            state,
            question,
            client_instance=CLIENT,
            model="gpt-4o-mini",
            temperature=0.0,
        )
    except Exception as exc:
        return {
            "id": test_id,
            "category": test_case["category"],
            "question": question,
            "status": "ERROR",
            "response": str(exc),
            "failures": [f"Exception: {exc}"],
        }

    response_lower = response.lower()
    failures = []

    # Check must_contain_any
    if test_case.get("must_contain_any"):
        found_any = False
        for phrase in test_case["must_contain_any"]:
            if phrase.lower() in response_lower:
                found_any = True
                break
        if not found_any:
            failures.append(
                f"MISSING: Expected at least one of: {test_case['must_contain_any']}"
            )

    # Check must_not_contain
    for phrase in test_case.get("must_not_contain", []):
        if phrase.lower() in response_lower:
            failures.append(f"FORBIDDEN PHRASE FOUND: '{phrase}'")

    # Check global forbidden phrases (info.json leakage)
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in response_lower:
            failures.append(f"INFO.JSON LEAKAGE: '{phrase}'")

    # Check CTA presence
    if "<a href=" not in response:
        failures.append("MISSING CTA: No HTML <a> link found in response")

    status = "PASS" if not failures else "FAIL"

    return {
        "id": test_id,
        "category": test_case["category"],
        "question": question,
        "status": status,
        "response": response,
        "failures": failures,
    }


def main():
    print("\n" + "=" * 70)
    print("  SENIOR QA ENGINEER TEST SUITE — Fast Sales AI Chatbot")
    print("  PDF-Only Compliance Testing")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []
    pass_count = 0
    fail_count = 0
    error_count = 0

    total = len(TEST_CASES)

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{total}] Testing {tc['id']}: {tc['question'][:60]}...")

        result = run_single_test(tc)
        results.append(result)

        if result["status"] == "PASS":
            pass_count += 1
            print(f"  ✅ PASS")
        elif result["status"] == "FAIL":
            fail_count += 1
            print(f"  ❌ FAIL")
            for f in result["failures"]:
                print(f"     → {f}")
        else:
            error_count += 1
            print(f"  ⚠️  ERROR: {result['failures'][0]}")

        # Rate limit protection
        time.sleep(1.5)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total Tests : {total}")
    print(f"  ✅ Passed   : {pass_count}")
    print(f"  ❌ Failed   : {fail_count}")
    print(f"  ⚠️  Errors   : {error_count}")
    print(f"  Pass Rate   : {pass_count/total*100:.1f}%")
    print("=" * 70)

    # ── Detailed Failure Report ───────────────────────────────────────────
    failed_results = [r for r in results if r["status"] != "PASS"]
    if failed_results:
        print("\n" + "─" * 70)
        print("  DETAILED FAILURE REPORT")
        print("─" * 70)
        for r in failed_results:
            print(f"\n  [{r['id']}] {r['category']}")
            print(f"  Q: {r['question']}")
            print(f"  Status: {r['status']}")
            for f in r["failures"]:
                print(f"  ❌ {f}")
            print(f"  Response (first 300 chars):")
            print(f"    {r['response'][:300]}...")

    # ── Save Full Results to JSON ─────────────────────────────────────────
    output_path = Path(__file__).resolve().parent / "qa_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_date": datetime.now().isoformat(),
                "total": total,
                "passed": pass_count,
                "failed": fail_count,
                "errors": error_count,
                "pass_rate": f"{pass_count/total*100:.1f}%",
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n  📄 Full results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
