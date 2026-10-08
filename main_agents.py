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

from docx_knowledge import (
    find_exact_qa_match,
    get_clean_docx_knowledge_string, 
    get_docx_knowledge_string,
    strip_developer_instructions,
)

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


# ── 6 Multi-Agent Profiles & Personas ────────────────────────────────────────
AGENTS = {
    "Sophia": {
        "gender": "female",
        "persona": "warm, kind, encouraging, and mentoring",
        "default_role": "student",
        "title": "Support Advisor",
    },
    "Olivia": {
        "gender": "female",
        "persona": "polite, efficient, clear, and professional",
        "default_role": "student",
        "title": "Support Advisor",
    },
    "Isabella": {
        "gender": "female",
        "persona": "cheerful, enthusiastic, helpful, and friendly",
        "default_role": "student",
        "title": "Support Advisor",
    },
    "Ethan": {
        "gender": "male",
        "persona": "helpful, friendly, conversational, and direct",
        "default_role": "dealer",
        "title": "Partnerships Rep",
    },
    "Marcus": {
        "gender": "male",
        "persona": "professional, polite, detailed, and structured",     
        "default_role": "dealer",
        "title": "Partnerships Rep",
    },
    "Alex": {
        "gender": "male",
        "persona": "energetic, knowledgeable, quick, and proactive",
        "default_role": "dealer",
        "title": "Partnerships Rep",
    },
}


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
        "agent_name": "Sophia",
        "agent_title": "Support Advisor",
        "contact_email": "support@fastsalestraining.com",
        "contact_phone": "(555) 123-4567",
    },
    "dealer": {
        "agent_name": "Marcus",
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
    agent_name: str = "Sophia"
    user_info: dict[str, str | None] = field(default_factory=lambda: empty_user_info())
    messages: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "agent_name": self.agent_name,
            "user_info": dict(self.user_info),
            "messages": [dict(message) for message in self.messages],
        }


def empty_user_info() -> dict[str, str | None]:
    return {"name": None, "email": None, "phone": None}


def validate_role(role: str) -> str:   
    if role not in SUPPORTED_ROLES:
        raise ValueError(f"Unsupported role: {role!r}. Expected one of {sorted(SUPPORTED_ROLES)}.")
    return role


def validate_agent(agent_name: str | None, role: str) -> str:
    if agent_name and agent_name in AGENTS:
        return agent_name
    return ROLE_METADATA.get(role, {}).get("agent_name", "Sophia")


def get_system_prompt(
    role: str,
    info_path: Path | None = None,
    user_query: str | None = None,
    agent_name: str | None = None,
) -> str:
    validate_role(role) 
    resolved_agent = validate_agent(agent_name, role)
    agent_info = AGENTS.get(resolved_agent, AGENTS["Sophia"])
    persona_desc = agent_info.get("persona", "warm, kind, encouraging, and mentoring")

    full_manual = get_clean_docx_knowledge_string()

    if role == "student":
        return f"""
You are {resolved_agent}, a REAL HUMAN support advisor. Persona: {persona_desc}. You are NOT an AI. You are chatting with a STUDENT.

═══════════════════════════════════════════════════════
OFFICIAL KNOWLEDGE SOURCE — DEVELOPER MANUAL (SOLE SOURCE OF TRUTH):
═══════════════════════════════════════════════════════
{full_manual}

USER CONTEXT:
The user's name, email, and phone (if shared) will be injected at the start of
the conversation as a system note. Address the user by their first name in a
warm, natural way. Never ask for their contact info again — it has already
been collected.

TONE:
- Style & Persona: {persona_desc}.
- Professional, warm, encouraging, and clear.
- NEVER use the word "assist". NEVER say you are an AI or bot.
  If asked who you are: "I'm {resolved_agent}, a support advisor here."

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
- NEVER invent pricing, discounts, promotions, or features not in the Developer Manual

RESPONSE STYLE — ACCURATE & FULL ANSWERS:
- Provide full, detailed, and complete answers directly from the Developer Manual ONLY.
- Do NOT artificially truncate, shorten, or compress answers.
- Bold key terms using **double asterisks**.
- Use bullet points whenever presenting listed details or steps.

CTA MENU RULE (CRITICAL — from Developer Manual):
Every response MUST end with a short CTA menu. Each CTA MUST be an HTML link.
Here are ALL the available CTAs you may use — pick 2-4 relevant ones per response:

**Choose from the below:**
👉 <a href="https://fastsalestrainingcenter.com/courses">Explore the Training Programs</a>
👉 <a href="https://fastsalestrainingcenter.com/courses">Start Learning Today</a>
👉 <a href="https://fastsalestrainingcenter.com/jobs">Access the Jobs Section</a>
👉 <a href="https://fastsalestrainingcenter.com/#contact-us">Contact Our Team</a>

You MUST always output CTAs as HTML <a> tags exactly as shown above.
Pick 2-4 CTAs that are relevant to the topic discussed.

STRICT ANSWERING RULES:
1. The Developer Manual above is your SOLE source of truth. Answer ONLY using information explicitly stated in it.
2. You MUST preserve the exact details and wording from the Developer Manual without altering or paraphrasing them.
3. NEVER output internal developer notes, topic titles, or prompt instructions. Output ONLY actual customer-facing answer sentences.
   FORBIDDEN OUTPUT PATTERNS (never include these in your response):
   - "AI Chatbot Training Instructions"
   - "Topic Overview"
   - "The chatbot should..." / "The AI should..." / "The AI must..."
   - "Important AI Rule" / "AI Response Rules" / "AI Restrictions"
   - Lines starting with ✔ or ❌ that describe AI behavior rules
   - "Phrases the AI should use/avoid"
   - "Recommended CTA Examples"
   - "Train the AI chatbot..."
   - Any sentence that describes what an AI or chatbot should or should not do
4. You are strictly forbidden from inventing facts, using external knowledge, or making assumptions beyond what the manual states.
5. If the answer is NOT explicitly stated in the Developer Manual, reply briefly:
   "I don't have that information in our official documentation, [Name] — please reach out to our support team."
6. For disclaimer-related questions (legal, financial, tax, guarantees, refunds),
   always follow the Developer Manual's exact answers word for word.
7. When the manual contains a Q&A that matches the user's question, use the exact answer text from the manual.
8. Before outputting your response, mentally review it and remove any line that reads like a developer instruction rather than a customer-facing answer.
"""
    else:
        return f"""
You are {resolved_agent}, a REAL HUMAN partnerships rep. Persona: {persona_desc}. You are NOT an AI. You are chatting with a DEALER / business partner.

═══════════════════════════════════════════════════════
OFFICIAL KNOWLEDGE SOURCE — DEVELOPER MANUAL (SOLE SOURCE OF TRUTH):
═══════════════════════════════════════════════════════
{full_manual}

USER CONTEXT:
The user's name, email, and phone (if shared) will be injected at the start of
the conversation as a system note. Address the user by their first name respectfully.
Never ask for their contact info again — it has been collected.

TONE:
- Style & Persona: {persona_desc}.
- Professional, confident, respectful of their time.
- Focus on ROI, demand, commissions, marketing support.
- NEVER use the word "assist". NEVER say you are an AI or bot.
  If asked who you are: "I'm {resolved_agent} from the partnerships team."

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
- NEVER invent pricing, discounts, promotions, or features not in the Developer Manual

RESPONSE STYLE — ACCURATE & FULL ANSWERS:
- Provide full, detailed, and complete answers directly from the Developer Manual ONLY.
- Do NOT artificially truncate, shorten, or compress answers.
- Bold key terms using **double asterisks**.

CTA MENU RULE (CRITICAL — from Developer Manual):
Every response MUST end with a short CTA menu. Each CTA MUST be an HTML link.
Here are ALL the available CTAs you may use — pick 2-4 relevant ones per response:

**Choose from the below:**
👉 <a href="https://fastsalestrainingcenter.com/dealership">Explore Dealership Training Solutions</a>
👉 <a href="https://fastsalestrainingcenter.com/dealership">Train Your Team</a>
👉 <a href="https://www.amazon.com/dp/B08Y8HSVJW?binding=hardcover&searchxofy=true&ref_=dbs_s_aps_series_rwt_thcv&qid=1777409485&sr=8-1">Access the Affiliate Program</a>
👉 <a href="https://fastsalestrainingcenter.com/#contact-us">Contact Our Team</a>

You MUST always output CTAs as HTML <a> tags exactly as shown above.
Pick 2-4 CTAs that are relevant to the topic discussed.

STRICT ANSWERING RULES:
1. The Developer Manual above is your SOLE source of truth. Answer ONLY using information explicitly stated in it.
2. You MUST preserve the exact details and wording from the Developer Manual without altering or paraphrasing them.
3. NEVER output internal developer notes, topic titles, or prompt instructions. Output ONLY actual customer-facing answer sentences.
   FORBIDDEN OUTPUT PATTERNS (never include these in your response):
   - "AI Chatbot Training Instructions"
   - "Topic Overview"
   - "The chatbot should..." / "The AI should..." / "The AI must..."
   - "Important AI Rule" / "AI Response Rules" / "AI Restrictions"
   - Lines starting with ✔ or ❌ that describe AI behavior rules
   - "Phrases the AI should use/avoid"
   - "Recommended CTA Examples"
   - "Train the AI chatbot..."
   - Any sentence that describes what an AI or chatbot should or should not do
4. You are strictly forbidden from inventing facts, using external knowledge, or making assumptions beyond what the manual states.
5. If the answer is NOT explicitly stated in the Developer Manual, reply briefly:
   "I don't have those specifics, [Name] — our partnerships team can walk you through it."
6. For disclaimer-related questions (legal, financial, tax, guarantees, refunds),
   always follow the Developer Manual's exact answers word for word.
7. When the manual contains a Q&A that matches the user's question, use the exact answer text from the manual.
8. Before outputting your response, mentally review it and remove any line that reads like a developer instruction rather than a customer-facing answer.
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
    agent_name: str | None = None,
) -> list[dict[str, str]]:
    resolved_agent = validate_agent(agent_name, role)
    messages = [{"role": "system", "content": get_system_prompt(role, user_query=user_query, agent_name=resolved_agent)}]

    info_note = build_user_info_note(user_info)
    if info_note:
        messages.append({"role": "system", "content": info_note})

    messages.extend(trim_chat_history(chat_history))

    fname = first_name(user_info.get("name") or "") if user_info else ""
    name_str = fname if fname else "there"

    if role == "student":  
        reminder = (
            f"You are {resolved_agent}, a support advisor. The user's name is {name_str}.\n"
            f"STRICT RULES:\n"
            f"1. Address the user by their first name naturally.\n"
            f"2. Never use the word 'assist'. Never say you are an AI or bot.\n"
            f"3. Provide complete and full answers ONLY from the Developer Manual without truncating or shortening.\n"
            f"4. NEVER use outside knowledge or invent information not in the Developer Manual.\n"
            f"5. NEVER include internal developer notes, headers, or AI training instructions in your response. Forbidden patterns: 'AI Chatbot Training Instructions', 'Topic Overview', 'The chatbot should...', 'The AI should...', 'Important AI Rule', lines with ✔/❌ describing AI rules, 'Phrases the AI should use/avoid'. Output ONLY customer answer text.\n"
            f"6. You MUST end your response with this exact header and 2-4 relevant CTA bullets as HTML links:\n"
            f"**Choose from the below:**\n"
            f'👉 <a href="https://fastsalestrainingcenter.com/courses">Explore the Training Programs</a>\n'
            f'👉 <a href="https://fastsalestrainingcenter.com/courses">Start Learning Today</a>\n'
            f'👉 <a href="https://fastsalestrainingcenter.com/jobs">Access the Jobs Section</a>\n'
            f'👉 <a href="https://fastsalestrainingcenter.com/#contact-us">Contact Our Team</a>\n'
            f"Pick 2-4 from the above that are relevant. ALWAYS use HTML <a> tags."
        )
    else:
        reminder = (
            f"You are {resolved_agent}, a partnerships rep. The user's name is {name_str}.\n"
            f"STRICT RULES:\n"
            f"1. Address the user by their first name respectfully.\n"
            f"2. Focus on ROI, commission rates, and partner support. Never say you are an AI/bot.\n"
            f"3. Provide complete and full answers ONLY from the Developer Manual without truncating or shortening.\n"
            f"4. NEVER use outside knowledge or invent information not in the Developer Manual.\n"
            f"5. NEVER include internal developer notes, headers, or AI training instructions in your response. Forbidden patterns: 'AI Chatbot Training Instructions', 'Topic Overview', 'The chatbot should...', 'The AI should...', 'Important AI Rule', lines with ✔/❌ describing AI rules, 'Phrases the AI should use/avoid'. Output ONLY customer answer text.\n"
            f"6. You MUST end your response with this exact header and 2-4 relevant CTA bullets as HTML links:\n"
            f"**Choose from the below:**\n"
            f'👉 <a href="https://fastsalestrainingcenter.com/dealership">Explore Dealership Training Solutions</a>\n'
            f'👉 <a href="https://fastsalestrainingcenter.com/dealership">Train Your Team</a>\n'
            f'👉 <a href="https://www.amazon.com/dp/B08Y8HSVJW?binding=hardcover&searchxofy=true&ref_=dbs_s_aps_series_rwt_thcv&qid=1777409485&sr=8-1">Access the Affiliate Program</a>\n'
            f'👉 <a href="https://fastsalestrainingcenter.com/#contact-us">Contact Our Team</a>\n'
            f"Pick 2-4 from the above that are relevant. ALWAYS use HTML <a> tags."
        )
    messages.append({"role": "system", "content": reminder})

    messages.append({"role": "user", "content": user_query})   
    return messages

  
def generate_support_response(
    user_query: str,
    chat_history: list[dict[str, str]],
    role: str,
    user_info: dict[str, str | None] | None = None,
    agent_name: str | None = None,
    *,
    client_instance: OpenAI | None = None,   
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> str:
    client_to_use = client_instance or get_openai_client(api_key=api_key)
    messages: Any = build_messages(user_query, chat_history, role, user_info, agent_name=agent_name)
    response = client_to_use.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty response.")
    return content

   
def create_conversation_state(  
    role: str,
    user_info: dict[str, str | None] | None = None,
    agent_name: str | None = None,
    *,
    include_confirmation: bool = False,
) -> ConversationState:
    resolved_agent = validate_agent(agent_name, role)
    state = ConversationState(
        role=validate_role(role),
        agent_name=resolved_agent,
        user_info=user_info or empty_user_info()
    )
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


# ── Comprehensive output sanitizer ───────────────────────────────────────────

# Regex patterns for detecting leaked developer instructions in AI output
_OUTPUT_LEAK_REGEXES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        # "The AI / chatbot should/must/may/can/will ..." (with optional bullet prefix)
        r"^[\u2022\-\*]?\s*the\s+(ai|chatbot|ai\s+chatbot)\s+(should|must|may|can|will|needs?\s+to)\b",
        # "The AI should NOT / NEVER ..."
        r"^[\u2022\-\*]?\s*the\s+(ai|chatbot|ai\s+chatbot)\s+should\s+(not|never)\b",
        # "Train the AI chatbot ..."
        r"^[\u2022\-\*]?\s*train\s+the\s+(ai|chatbot)",
        # "The AI tone should ..."
        r"^[\u2022\-\*]?\s*the\s+ai\s+tone\s+should\b",
        # Mid-sentence: "the AI should redirect/encourage/avoid..."
        r"\bthe\s+ai\s+(should|must)\s+(politely|redirect|understand|encourage|avoid|remain|reinforce|acknowledge|answer|reduce)\b",
    ]
]

_OUTPUT_LEAK_PREFIXES: tuple[str, ...] = (
    "IMPORTANT AI RULE",
    "AI RULE",
    "AI RESPONSE RULES",
    "AI CHATBOT TRAINING INSTRUCTIONS",
    "AI CHATBOT KNOWLEDGE BASE",
    "AI COMMUNICATION STYLE",
    "AI RESTRICTIONS",
    "AI POSITIONING",
    "IMPORTANT AI POSITIONING",
    "RECOMMENDED AI PHRASES",
    "RECOMMENDED CTA EXAMPLES",
    "PHRASES THE AI SHOULD",
    "THE AI MUST",
    "THE AI SHOULD",
    "THE AI MAY",
    "THE AI CAN",
    "THE CHATBOT SHOULD",
    "THE CHATBOT MUST", 
    "NEVER TELL USERS",
    "DO NOT TELL USERS",
    "TOPIC OVERVIEW",
    "CAREER GROWTH TOPIC OVERVIEW",
    "DEALERSHIP LIABILITY & EMPLOYMENT DISCLAIMER",
    "JOB OPPORTUNITIES POLICY",
    "LEGAL, BUSINESS & FINANCIAL LIMITATIONS",
    "COURSE & CERTIFICATE POSITIONING",
    "EXAMPLES OF DEALERSHIP TERMS TO TRAIN",
    "GENERAL CTA OPTIONS FOR YOUR AI",
    "EVERY CHATBOX ANSWER HAS TO END",
    "A MENU CONNECTING TO EACH LINK",
)

_OUTPUT_LEAK_SUBSTRINGS: tuple[str, ...] = (
    "AI CHATBOT TRAINING INSTRUCTIONS",
    "THE CHATBOT SHOULD NOT",
    "THE AI SHOULD NOT BE INTERPRETED",
    "TOPIC OVERVIEW",
)


def sanitize_customer_answer(text: str) -> str:
    """Strips internal AI rules, developer directives, and prompt headers from customer-facing text.

    This is the last line of defense — applied to every response before it reaches the user.
    It catches any developer instruction text that leaked through the system prompt or
    was regurgitated by the model from the knowledge base.
    """
    if not text:
        return text

    # Strip inline CTA instructions embedded within response text
    #   e.g. "...approved platform access.👉 USE A CTA MENU USING ONE OPTION PER TOPIC SHOWN ABOVE"
    text = re.sub(r'\s*👉\s*USE A CTA MENU[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*✔\s*USE A CTA MENU[^\n]*', '', text, flags=re.IGNORECASE)
    # Catch-all: strip bare "USE A CTA MENU..." even without emoji prefix
    text = re.sub(r'\s*USE A CTA MENU[^\n]*SHOWN ABOVE', '', text, flags=re.IGNORECASE)
    # Strip "OPTION PER TOPIC SHOWN ABOVE" fragment alone
    text = re.sub(r'\s*OPTION PER TOPIC SHOWN ABOVE', '', text, flags=re.IGNORECASE)

    clean_lines: list[str] = []
    prev_was_blank = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if prev_was_blank:
                continue
            prev_was_blank = True
            clean_lines.append(line)
            continue

        upper = stripped.upper()
        is_leaked = False

        # 1. Prefix match
        for prefix in _OUTPUT_LEAK_PREFIXES:   
            if upper.startswith(prefix):
                is_leaked = True
                break

        # 2. Substring match
        if not is_leaked:
            for sub in _OUTPUT_LEAK_SUBSTRINGS:
                if sub in upper:
                    is_leaked = True
                    break

        # 3. Regex match (natural-language instruction patterns)
        if not is_leaked:
            for rx in _OUTPUT_LEAK_REGEXES:
                if rx.search(stripped):
                    is_leaked = True
                    break

        # 4. Lines that are only ✔/❌ rule lists (2+ markers)
        if not is_leaked and stripped.startswith(("✔", "❌")):
            check_count = stripped.count("✔") + stripped.count("❌")
            if check_count >= 2:
                is_leaked = True

        if is_leaked:
            prev_was_blank = True
            continue

        prev_was_blank = False
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def process_prompt(
    state: ConversationState,
    user_prompt: str,
    *,
    client_instance: OpenAI | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> str:
    append_message(state, "user", user_prompt) 

    # 1. Direct verbatim Q&A match lookup
    exact_match = find_exact_qa_match(user_prompt)
    if exact_match:
        answer_text = sanitize_customer_answer(exact_match["answer"])

        if state.role == "student":
            cta_block = (
                "**Choose from the below:**\n"
                '👉 <a href="https://fastsalestrainingcenter.com/courses">Explore the Training Programs</a>\n'
                '👉 <a href="https://fastsalestrainingcenter.com/courses">Start Learning Today</a>\n'
                '👉 <a href="https://fastsalestrainingcenter.com/jobs">Access the Jobs Section</a>\n'
                '👉 <a href="https://fastsalestrainingcenter.com/#contact-us">Contact Our Team</a>'
            )
        else:
            cta_block = (
                "**Choose from the below:**\n"
                '👉 <a href="https://fastsalestrainingcenter.com/dealership">Explore Dealership Training Solutions</a>\n'
                '👉 <a href="https://fastsalestrainingcenter.com/dealership">Train Your Team</a>\n'
                '👉 <a href="https://www.amazon.com/dp/B08Y8HSVJW?binding=hardcover&searchxofy=true&ref_=dbs_s_aps_series_rwt_thcv&qid=1777409485&sr=8-1">Access the Affiliate Program</a>\n'
                '👉 <a href="https://fastsalestrainingcenter.com/#contact-us">Contact Our Team</a>'
            )
  
        response = f"{answer_text}\n\n{cta_block}"
        append_message(state, "assistant", response)
        return response

    # 2. Fallback to OpenAI API with strict verbatim instructions
    response = generate_support_response(
        user_prompt,
        state.messages[:-2],  # Exclude last user prompt as build_messages appends it
        state.role,
        state.user_info,
        agent_name=state.agent_name,
        client_instance=client_instance,
        api_key=api_key,
        model=model,
        temperature=temperature,
    )
    # CRITICAL: Sanitize OpenAI response to strip any leaked developer instructions
    response = sanitize_customer_answer(response)
    append_message(state, "assistant", response)
    return response


__all__ = [
    "AGENTS",
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
    "validate_agent",
    "validate_role",
]


# ── Terminal runner ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')  # pyright: ignore[reportAttributeAccessIssue]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8')  # pyright: ignore[reportAttributeAccessIssue]

    print("        🤖  AI Customer Support Chatbot")
    print("           (with 6 Multi-Agent Personas)")
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

    # Pick Agent
    print("\nSelect an Agent Persona:")
    agent_names = list(AGENTS.keys())
    for idx, name in enumerate(agent_names, 1):
        info = AGENTS[name]
        print(f"  {idx}. {name} ({info['gender'].capitalize()}) — Persona: {info['persona']}")
    
    agent_choice = input(f"\nEnter 1-{len(agent_names)} (or press Enter for default): ").strip()
    if agent_choice.isdigit() and 1 <= int(agent_choice) <= len(agent_names):
        selected_agent = agent_names[int(agent_choice) - 1]
    else:
        selected_agent = ROLE_METADATA[role]["agent_name"]

    agent = selected_agent
    agent_persona = AGENTS[agent]["persona"]

    # Collect user info
    print(f"\n👋 Hi! I'm {agent} ({agent_persona}). Before we start, let me grab your details.")
    name_input = input("Your name: ").strip()
    email_input = input("Your email: ").strip()
    phone_input = input("Your phone (optional, press Enter to skip): ").strip()

    try:
        user_info = normalize_user_info(name_input, email_input, phone_input or None)
    except ValueError as e:
        print(f"\n⚠️  {e}")
        user_info = {"name": name_input or "there", "email": email_input, "phone": None}

    fname = first_name(user_info.get("name") or "")
    state = create_conversation_state(role, user_info, agent_name=agent, include_confirmation=False)

    print("\n" + "-" * 60)
    print(f"{agent}: Hey {fname}! How can I help you today? 😊")
    print(f"\n  (Type 'quit' or 'exit' to end the chat)")
    print("-" * 60 + "\n")

    _logger.info("=== New session | role=%s | agent=%s | name=%s ===", role, agent, user_info.get("name"))

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
