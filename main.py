from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from docx_knowledge import get_docx_knowledge_string

load_dotenv()

INFO_JSON_PATH = Path(__file__).resolve().with_name("info.json")
MAX_HISTORY_PAIRS = 10

# ── Logging setup (writes to chat_logs/ next to this file) ──────────────────
_LOG_DIR = Path(__file__).resolve().parent / "chat_logs"
_LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(_LOG_DIR / f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_logger = logging.getLogger(__name__)


def load_website_data(info_path: Path | None = None) -> dict:
    """Read and return the parsed contents of info.json."""
    path = info_path or INFO_JSON_PATH
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"info.json not found at: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"info.json contains invalid JSON: {exc}") from exc


def get_knowledge_base_string(info_path: Path | None = None) -> str:
    """Return a compact JSON string of info.json for use in system prompts."""
    data = load_website_data(info_path)
    return json.dumps(data, indent=2)


SUPPORTED_ROLES = {"student", "dealer"}

QUICK_REPLIES = {
    "student": [
        "What courses do you offer?",
        "I'm new — where do I start?",
        "Tell me about the available courses",
        "How do I find a job after training?",
        "What membership plans are available?",
    ],
    "dealer": [
        "How does the affiliate program work?",
        "Which courses have the highest demand?",
        "What marketing support do you provide?",
        "Tell me about job listings",
        "How do I become a partner?",
    ],
}

ROLE_METADATA = {
    "student": {
        "agent_name": "Sarah",
        "agent_title": "Support Advisor",
        "contact_email": "support@fastsalestraining.com",
        "contact_phone": "(555) 123-4567",
    },
    "dealer": {
        "agent_name": "Michael",
        "agent_title": "Partnerships Rep",
        "contact_email": "partners@fastsalestraining.com",
        "contact_phone": "(555) 123-4568",
    },
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{6,}\d)")
SKIP_WORDS = {"skip", "no", "n/a", "na", "none", "nope", "later", "no thanks", "no thank you"}


@dataclass(slots=True)
class ConversationState:
    role: str
    user_info: dict[str, str | None] = field(default_factory=lambda: empty_user_info())
    messages: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "user_info": dict(self.user_info),
            "messages": [dict(message) for message in self.messages],
        }


def empty_user_info() -> dict[str, str | None]:
    return {"name": None, "email": None, "phone": None}


def validate_role(role: str) -> str:
    if role not in SUPPORTED_ROLES:
        raise ValueError(f"Unsupported role: {role!r}. Expected one of {sorted(SUPPORTED_ROLES)}.")
    return role


def get_system_prompt(role: str, info_path: Path | None = None) -> str:
    validate_role(role)
    kb = get_knowledge_base_string(info_path)
    docx_kb = get_docx_knowledge_string()

    if role == "student":
        return f"""
You are Sarah, a REAL HUMAN support advisor. You are NOT an AI. You are chatting with a STUDENT.

═══════════════════════════════════════════════════════
DATABASE 1 — WEBSITE DATA (live from info.json):
═══════════════════════════════════════════════════════
{kb}

═══════════════════════════════════════════════════════
DATABASE 2 — DEVELOPER MANUAL (official chatbot training document):
═══════════════════════════════════════════════════════
{docx_kb}

USER CONTEXT:
The user's name, email, and phone (if shared) will be injected at the start of
the conversation as a system note. ALWAYS address the user by their first name
in a warm, natural way (e.g., "Sure, John!"). Never ask for their contact info
again — it has already been collected.

TONE:
- Warm, kind, encouraging — like a caring mentor.
- Use the user's first name naturally. No "sir"/"ma'am" for students.
- Use contractions, sound human, never robotic.
- NEVER use the word "assist". NEVER say you are an AI or bot.
  If asked who you are: "I'm Sarah, a support advisor here."

PHRASES TO AVOID (from Developer Manual):
- NEVER say "get rich", "guaranteed job", "guaranteed income",
  "become an expert instantly", "life-changing results"
- NEVER use overhyped or misleading sales language

AI RESTRICTIONS (from Developer Manual):
- NEVER promise employment, interviews, or promotions
- NEVER guarantee income, sales performance, or business results
- NEVER provide legal, financial, tax, or accounting advice
- NEVER misrepresent certifications
- NEVER claim dealership partnerships unless officially confirmed
- NEVER invent pricing, discounts, promotions, or features not listed in the databases

RESPONSE STYLE — KEEP IT SHORT AND PUNCHY:
- Default length: 1–3 sentences. Maximum 4 sentences unless the user asks
  for "details", "more info", a "list", or "everything".
- Bold only the key term using **double asterisks**.
- Bullet points ONLY when listing 3+ concrete items the user asked for.
- No filler ("Great question!", "I'd be happy to..."). Get straight to value.
- One emoji max per reply, only if it adds warmth.

SMART FOLLOW-UP RULE:
- If the user is VAGUE (e.g. "I want a course", "help me", "what do you offer"),
  ask ONE short, targeted question to understand their goal BEFORE listing options.
- After ONE clarifying question, commit to a recommendation — do NOT keep asking.
- If the user's question is already specific, answer it directly with NO follow-up.
- NEVER ask more than ONE question in a single reply. Never chain questions.
- Do NOT list all courses unless the user explicitly asks "what courses do you have".

CTA MENU RULE (CRITICAL — from Developer Manual):
Every response MUST end with a short CTA menu. Each CTA MUST be an HTML link.
Here are ALL the available CTAs you may use — pick 2-4 relevant ones per response:

**Choose from the below:**
👉 <a href="https://fast-selles-training-center.vercel.app/courses">Explore the Training Programs</a>
👉 <a href="https://fast-selles-training-center.vercel.app/courses">Start Learning Today</a>
👉 <a href="https://fast-selles-training-center.vercel.app/jobs">Access the Jobs Section</a>
👉 <a href="https://fast-selles-training-center.vercel.app/#contact-us">Contact Our Team</a>

You MUST always output CTAs as HTML <a> tags exactly as shown above.
Pick 2-4 CTAs that are relevant to the topic discussed.
Do NOT repeat the same CTAs every time — vary them based on context.

INSTRUCTIONS:
1. Answer using BOTH databases. If the Developer Manual has a suggested answer
   for the user's question, use that answer (or paraphrase it naturally).
2. Never invent facts not in either database.
3. Naturally suggest a related course, membership, or job only when directly relevant.
4. If the answer is NOT in either database, reply briefly:
   "I don't have that on hand, [Name] — please reach out to our support team."
5. For disclaimer-related questions (legal, financial, tax, guarantees, refunds),
   always follow the Developer Manual's answers and include:
   👉 View Full Disclaimer
"""
    else:
        return f"""
You are Michael, a REAL HUMAN partnerships rep. You are NOT an AI. You are chatting with a DEALER / business partner.

═══════════════════════════════════════════════════════
DATABASE 1 — WEBSITE DATA (live from info.json):
═══════════════════════════════════════════════════════
{kb}

═══════════════════════════════════════════════════════
DATABASE 2 — DEVELOPER MANUAL (official chatbot training document):
═══════════════════════════════════════════════════════
{docx_kb}

USER CONTEXT:
The user's name, email, and phone (if shared) will be injected at the start of
the conversation as a system note. ALWAYS address the user by their first name
respectfully. Never ask for their contact info again — it has been collected.

TONE:
- Professional, confident, respectful of their time.
- Use their first name. If gender is unclear, default to a polite "sir".
- Focus on ROI, demand, commissions, marketing support.
- NEVER use the word "assist". NEVER say you are an AI or bot.
  If asked who you are: "I'm Michael from the partnerships team."

PHRASES TO AVOID (from Developer Manual):
- NEVER say "get rich", "guaranteed job", "guaranteed income",
  "become an expert instantly", "life-changing results"
- NEVER use overhyped or misleading sales language

AI RESTRICTIONS (from Developer Manual):
- NEVER promise employment, interviews, or promotions
- NEVER guarantee income, sales performance, or business results
- NEVER provide legal, financial, tax, or accounting advice
- NEVER misrepresent certifications
- NEVER claim dealership partnerships unless officially confirmed
- NEVER invent pricing, discounts, promotions, or features not listed in the databases

RESPONSE STYLE — KEEP IT SHORT AND PUNCHY:
- Default length: 1–3 sentences. Maximum 4 sentences unless the user asks
  for "details", "breakdown", or "everything".
- Bold key numbers/benefits with **double asterisks**.
- Bullet points ONLY when listing 3+ concrete items the user asked for.
- No filler ("Great question!", "Absolutely happy to..."). Lead with value.

SMART FOLLOW-UP RULE:
- If the dealer is VAGUE (e.g. "tell me about your programs", "I want to partner"),
  ask ONE short, targeted question to understand their goal BEFORE explaining.
- After ONE clarifying question, commit to an answer — do NOT keep asking.
- If the question is already specific, answer it directly with NO follow-up.
- NEVER ask more than ONE question in a single reply. Never chain questions.

CTA MENU RULE (CRITICAL — from Developer Manual):
Every response MUST end with a short CTA menu. Each CTA MUST be an HTML link.
Here are ALL the available CTAs you may use — pick 2-4 relevant ones per response:

**Choose from the below:**
👉 <a href="https://fast-selles-training-center.vercel.app/dealership">Explore Dealership Training Solutions</a>
👉 <a href="https://fast-selles-training-center.vercel.app/dealership">Train Your Team</a>
👉 <a href="https://www.amazon.com/dp/B08Y8HSVJW?binding=hardcover&searchxofy=true&ref_=dbs_s_aps_series_rwt_thcv&qid=1777409485&sr=8-1">Access the Affiliate Program</a>
👉 <a href="https://fast-selles-training-center.vercel.app/#contact-us">Contact Our Team</a>

You MUST always output CTAs as HTML <a> tags exactly as shown above.
Pick 2-4 CTAs that are relevant to the topic discussed.
Do NOT repeat the same CTAs every time — vary them based on context.

INSTRUCTIONS:
1. Answer using BOTH databases. If the Developer Manual has a suggested answer
   for the user's question, use that answer (or paraphrase it naturally).
2. Never invent facts not in either database.
3. Highlight relevant memberships, jobs, or courses only when relevant.
4. If the answer is NOT in either database, reply briefly:
   "I don't have those specifics, [Name] — our partnerships team can walk you through it."
5. For disclaimer-related questions (legal, financial, tax, guarantees, refunds),
   always follow the Developer Manual's answers and include:
   👉 View Full Disclaimer
"""


def get_openai_client(api_key: str | None = None) -> OpenAI:
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=resolved_api_key)


def extract_email(text: str) -> str | None:
    match = EMAIL_RE.search(text or "")
    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    match = PHONE_RE.search(text or "")
    return match.group(0).strip() if match else None


def extract_name(text: str) -> str:
    cleaned = (text or "").strip().strip(".!?")
    lowered = cleaned.lower()
    for prefix in ("my name is ", "i am ", "i'm ", "this is ", "it's ", "name is ", "call me "):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip().strip(".!?")
            break
    return " ".join(cleaned.split()[:4])


def first_name(full_name: str) -> str:
    return (full_name or "").split()[0] if full_name else ""


def normalize_user_info(
    name: str,
    email: str,
    phone: str | None = None,
) -> dict[str, str | None]:
    name_clean = name.strip()
    email_clean = extract_email(email.strip())
    phone_clean = extract_phone(phone.strip()) if phone and phone.strip() else None

    if len(name_clean) < 2:
        raise ValueError("Please enter your full name.")
    if not email_clean:
        raise ValueError("Please enter a valid email address.")

    return {
        "name": name_clean,
        "email": email_clean,
        "phone": phone_clean,
    }


def build_intake_confirmation(name: str) -> str:
    return f"Thanks, **{first_name(name)}**! We've received your info and you're all set."


def build_user_info_note(user_info: dict[str, str | None] | None) -> str | None:
    if not user_info or not user_info.get("name"):
        return None

    return (
        f"COLLECTED USER INFO — name: {user_info.get('name')}, "
        f"email: {user_info.get('email') or 'not provided'}, "
        f"phone: {user_info.get('phone') or 'not provided'}. "
        "Address them by their first name. Do NOT ask for these details again. "
        "IMPORTANT: If the user asks for their own contact details (email, phone, name), "
        "you MUST tell them exactly what is stored above."
    )


def trim_chat_history(chat_history: list[dict[str, str]], max_history_pairs: int = MAX_HISTORY_PAIRS) -> list[dict[str, str]]:
    return chat_history[-(max_history_pairs * 2):]


def build_messages(
    user_query: str,
    chat_history: list[dict[str, str]],
    role: str,
    user_info: dict[str, str | None] | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": get_system_prompt(role)}]

    info_note = build_user_info_note(user_info)
    if info_note:
        messages.append({"role": "system", "content": info_note})

    messages.extend(trim_chat_history(chat_history))

    fname = first_name(user_info.get("name") or "") if user_info else ""
    name_str = fname if fname else "there"

    if role == "student":
        reminder = (
            f"You are Sarah, a support advisor. The user's name is {name_str}.\n"
            f"STRICT RULES:\n"
            f"1. You MUST address the user by their first name (e.g. \"Sure, {name_str}!\").\n"
            f"2. Never use the word 'assist'. Never say you are an AI or bot. You are a real human.\n"
            f"3. Keep replies short (1-3 sentences, max 4). Bold only key terms.\n"
            f"4. You MUST end your response with this exact header and 2-4 relevant CTA bullets as HTML links:\n"
            f"**Choose from the below:**\n"
            f'👉 <a href="https://fast-selles-training-center.vercel.app/courses">Explore the Training Programs</a>\n'
            f'👉 <a href="https://fast-selles-training-center.vercel.app/courses">Start Learning Today</a>\n'
            f'👉 <a href="https://fast-selles-training-center.vercel.app/jobs">Access the Jobs Section</a>\n'
            f'👉 <a href="https://fast-selles-training-center.vercel.app/#contact-us">Contact Our Team</a>\n'
            f"Pick 2-4 from the above that are relevant. ALWAYS use HTML <a> tags.\n"
            f"5. IMPORTANT: For disclaimer-related questions (guarantees, employment, refunds, legal, or financial), you MUST include this exact CTA: 👉 View Full Disclaimer"
        )
    else:
        reminder = (
            f"You are Michael, a partnerships rep. The user's name is {name_str}.\n"
            f"STRICT RULES:\n"
            f"1. You MUST address the user by their first name (respectfully).\n"
            f"2. Focus on ROI, commission rates, and partner support. Never say you are an AI/bot.\n"
            f"3. Keep replies short (1-3 sentences). Bold key numbers/benefits.\n"
            f"4. You MUST end your response with this exact header and 2-4 relevant CTA bullets as HTML links:\n"
            f"**Choose from the below:**\n"
            f'👉 <a href="https://fast-selles-training-center.vercel.app/dealership">Explore Dealership Training Solutions</a>\n'
            f'👉 <a href="https://fast-selles-training-center.vercel.app/dealership">Train Your Team</a>\n'
            f'👉 <a href="https://www.amazon.com/dp/B08Y8HSVJW?binding=hardcover&searchxofy=true&ref_=dbs_s_aps_series_rwt_thcv&qid=1777409485&sr=8-1">Access the Affiliate Program</a>\n'
            f'👉 <a href="https://fast-selles-training-center.vercel.app/#contact-us">Contact Our Team</a>\n'
            f"Pick 2-4 from the above that are relevant. ALWAYS use HTML <a> tags.\n"
            f"5. IMPORTANT: For disclaimer-related questions (guarantees, employment, refunds, legal, or financial), you MUST include this exact CTA: 👉 View Full Disclaimer"
        )
    messages.append({"role": "system", "content": reminder})

    messages.append({"role": "user", "content": user_query})
    return messages


def generate_support_response(
    user_query: str,
    chat_history: list[dict[str, str]],
    role: str,
    user_info: dict[str, str | None] | None = None,
    *,
    client_instance: OpenAI | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> str:
    client_to_use = client_instance or get_openai_client(api_key=api_key)
    response = client_to_use.chat.completions.create(
        model=model,
        messages=build_messages(user_query, chat_history, role, user_info),
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty response.")
    return content


def create_conversation_state(
    role: str,
    user_info: dict[str, str | None] | None = None,
    *,
    include_confirmation: bool = False,
) -> ConversationState:
    state = ConversationState(role=validate_role(role), user_info=user_info or empty_user_info())
    if include_confirmation and state.user_info.get("name"):
        state.messages.append(
            {
                "role": "assistant",
                "content": build_intake_confirmation(state.user_info["name"] or ""),
            }
        )
    return state


def append_message(state: ConversationState, role: str, content: str) -> None:
    state.messages.append({"role": role, "content": content})


def process_prompt(
    state: ConversationState,
    user_prompt: str,
    *,
    client_instance: OpenAI | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> str:
    append_message(state, "user", user_prompt)
    response = generate_support_response(
        user_prompt,
        state.messages[:-1],
        state.role,
        state.user_info,
        client_instance=client_instance,
        api_key=api_key,
        model=model,
        temperature=temperature,
    )
    append_message(state, "assistant", response)
    return response


__all__ = [
    "ConversationState",
    "EMAIL_RE",
    "MAX_HISTORY_PAIRS",
    "PHONE_RE",
    "QUICK_REPLIES",
    "ROLE_METADATA",
    "SKIP_WORDS",
    "SUPPORTED_ROLES",
    "append_message",
    "build_intake_confirmation",
    "build_messages",
    "build_user_info_note",
    "create_conversation_state",
    "empty_user_info",
    "extract_email",
    "extract_name",
    "extract_phone",
    "first_name",
    "generate_support_response",
    "get_docx_knowledge_string",
    "get_knowledge_base_string",
    "get_openai_client",
    "get_system_prompt",
    "load_website_data",
    "normalize_user_info",
    "process_prompt",
    "trim_chat_history",
    "validate_role",
]


# ── Terminal runner ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("        🤖  AI Customer Support Chatbot")
    print("           (with Developer Manual knowledge)")
    print("=" * 60)

    # Show DOCX loading status
    docx_text = get_docx_knowledge_string()
    if docx_text:
        print(f"  ✅ Developer Manual loaded: {len(docx_text):,} chars")
    else:
        print("  ⚠️  Developer Manual not found — running without it")

    # Pick role
    print("\nAre you a:")
    print("  1. Student")
    print("  2. Dealer / Business Partner")
    while True:
        choice = input("\nEnter 1 or 2: ").strip()
        if choice == "1":
            role = "student"
            break
        elif choice == "2":
            role = "dealer"
            break
        print("Please enter 1 or 2.")

    meta = ROLE_METADATA[role]
    agent = meta["agent_name"]

    # Collect user info
    print(f"\n👋 Hi! I'm {agent}. Before we start, let me grab your details.")
    name_input = input("Your name: ").strip()
    email_input = input("Your email: ").strip()
    phone_input = input("Your phone (optional, press Enter to skip): ").strip()

    try:
        user_info = normalize_user_info(name_input, email_input, phone_input or None)
    except ValueError as e:
        print(f"\n⚠️  {e}")
        user_info = {"name": name_input or "there", "email": email_input, "phone": None}

    fname = first_name(user_info.get("name") or "")
    state = create_conversation_state(role, user_info, include_confirmation=False)

    print("\n" + "-" * 60)
    print(f"{agent}: Hey {fname}! How can I help you today? 😊")
    print(f"\n  (Type 'quit' or 'exit' to end the chat)")
    print("-" * 60 + "\n")

    _logger.info("=== New session | role=%s | name=%s ===", role, user_info.get("name"))

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{agent}: Goodbye, {fname}! Have a great day! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print(f"\n{agent}: Thanks for chatting, {fname}! Feel free to come back anytime. 👋\n")
            _logger.info("Session ended by user")
            break

        _logger.info("USER: %s", user_input)

        try:
            reply = process_prompt(state, user_input)
        except Exception as e:
            reply = f"Sorry, I ran into a technical issue. Please try again. (Error: {e})"

        print(f"\n{agent}: {reply}\n")
        _logger.info("BOT: %s", reply)

    _logger.info("=== Session ended | exchanges=%d ===", len(state.messages) // 2)
