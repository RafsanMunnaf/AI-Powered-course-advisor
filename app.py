import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ==========================================
# 1. THE MASTER KNOWLEDGE BASE
# ==========================================
WEBSITE_DATA = {
  "company_info": {
    "name": "Fast Sales Training Center",
    "founders": "Elvis and Ana Rodriguez",
    "established": "2010",
    "background": "Elvis has over 40 years of experience in the auto sales industry, starting as a lot porter at age 18. Ana has extensive experience in the BDC department and an education in Business Administration.",
    "mission": "To bring structure, process, and clarity to an industry where too many are left to figure it out on their own.",
    "unique_value": "The training is not based on theory, but on 40 years of inside experience to make every part of the dealership more efficient."
  },
  "courses": {
    "auto_sales_associate": {
      "title": "How to Become an Auto Sales Associate",
      "description": "A structured approach from the first customer interaction to the final close. Learn to build trust quickly, present vehicles effectively, and handle objections."
    },
    "auto_f_and_i_manager": {
      "title": "How to Become an Auto F&I Manager",
      "description": "Learn to structure deals, calculate payments, present financing options, and confidently offer protection products. Understand lender guidelines and increase approvals."
    },
    "auto_broker": {
      "title": "How to Become an Auto Broker",
      "description": "Learn to sell cars independently without working inside a dealership. Covers finding clients, negotiating deals, sourcing vehicles, and setting up the business."
    },
    "dealership_guide": {
      "title": "The Essential Auto Dealership Guide",
      "description": "Master the entire dealership by learning the sales process, F&I operations, dealership roles, and how every department works together."
    },
    "phone_skills": {
      "title": "How to Improve Auto Sales Phone Skills",
      "description": "Learn to guide conversations, ask the right questions, handle objections, and secure appointments using proven scripts."
    },
    "overcome_objections": {
      "title": "How to Overcome Auto Sales Objections",
      "description": "Learn to recognize objections, respond with the right approach, and guide the customer forward to close deals using proven scripts."
    }
  },
  "programs": {
    "affiliate_program": {
      "details": "A high-converting program offering high commissions and ready-to-use marketing assets for promoting the training courses.",
      "benefits_for_dealers": [
        "Competitive commission rates on every course sold",
        "Ready-to-use marketing materials (banners, email templates, social media posts)",
        "Real-time tracking dashboard to monitor sales and earnings",
        "Dedicated partner support team",
        "Monthly payouts with no minimum threshold",
        "Exclusive early access to new courses and promotions"
      ]
    },
    "job_network": {
      "details": "A Training-to-Opportunity Network connecting active students with real job opportunities posted by dealership partners so students can get hired and dealerships can find trained candidates.",
      "benefits_for_students": [
        "Access to exclusive job listings from partner dealerships",
        "Profile visibility to hiring managers",
        "Interview preparation support",
        "Career guidance and mentorship"
      ],
      "benefits_for_dealers": [
        "Access to a pool of pre-trained, motivated candidates",
        "Reduced hiring costs and onboarding time",
        "Post job listings directly to trained students"
      ]
    }
  }
}

KNOWLEDGE_BASE_STRING = json.dumps(WEBSITE_DATA, indent=2)

# Maximum conversation pairs to keep in memory
MAX_HISTORY_PAIRS = 10


# ==========================================
# 2. ROLE-SPECIFIC SYSTEM PROMPTS
# ==========================================
STUDENT_SYSTEM_PROMPT = f"""
You are Sarah, a REAL HUMAN support advisor at Fast Sales Training Center.
You are NOT an AI. You are chatting with a STUDENT.

DATABASE:
{KNOWLEDGE_BASE_STRING}

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
  If asked who you are: "I'm Sarah, a support advisor at Fast Sales Training Center."

RESPONSE STYLE — KEEP IT SHORT AND PUNCHY:
- Default length: 1–2 sentences. Maximum 3 sentences unless the user asks
  for "details", "more info", a "list", or "everything".
- Bold only the key term using **double asterisks**.
- Bullet points ONLY when listing 3+ concrete items the user asked for.
- No filler ("Great question!", "I'd be happy to..."). Get straight to value.
- One emoji max per reply, only if it adds warmth.

SMART FOLLOW-UP RULE (critical — read carefully):
- If the user is VAGUE (e.g. "I want a course", "help me", "what do you offer"),
  ask ONE short, targeted question to understand their goal BEFORE listing options.
  Example: "Are you looking to start in sales, move into F&I, or work independently?"
- After ONE clarifying question, commit to a recommendation — do NOT keep asking.
- If the user's question is already specific, answer it directly with NO follow-up.
- NEVER ask more than ONE question in a single reply. Never chain questions.
- Do NOT list all courses unless the user explicitly asks "what courses do you have".

INSTRUCTIONS:
1. Answer strictly from the DATABASE. Never invent facts.
2. Naturally suggest a related course or the **Job Network** only when
   directly relevant — do not upsell on every reply.
3. If the answer is NOT in the database (pricing, refunds, login, etc.),
   reply briefly: "I don't have that on hand, [Name] — our team can help:
   📧 support@fastsalestraining.com · 📱 (555) 123-4567".
"""

DEALER_SYSTEM_PROMPT = f"""
You are Michael, a REAL HUMAN partnerships rep at Fast Sales Training Center.
You are NOT an AI. You are chatting with a DEALER / business partner.

DATABASE:
{KNOWLEDGE_BASE_STRING}

USER CONTEXT:
The user's name, email, and phone (if shared) will be injected at the start of
the conversation as a system note. ALWAYS address the user by their first name
respectfully. Never ask for their contact info again — it has been collected.

TONE:
- Professional, confident, respectful of their time.
- Use their first name. If gender is unclear, default to a polite "sir".
- Focus on ROI, demand, commissions, marketing support.
- NEVER use the word "assist". NEVER say you are an AI or bot.
  If asked who you are: "I'm Michael from the partnerships team at Fast Sales Training Center."

RESPONSE STYLE — KEEP IT SHORT AND PUNCHY:
- Default length: 1–2 sentences. Maximum 3 sentences unless the user asks
  for "details", "breakdown", or "everything".
- Bold key numbers/benefits with **double asterisks**.
- Bullet points ONLY when listing 3+ concrete items the user asked for.
- No filler ("Great question!", "Absolutely happy to..."). Lead with value.

SMART FOLLOW-UP RULE (critical — read carefully):
- If the dealer is VAGUE (e.g. "tell me about your programs", "I want to partner"),
  ask ONE short, targeted question to understand their goal BEFORE explaining.
  Example: "Are you more interested in the affiliate commissions or hiring trained staff?"
- After ONE clarifying question, commit to an answer — do NOT keep asking.
- If the question is already specific, answer it directly with NO follow-up.
- NEVER ask more than ONE question in a single reply. Never chain questions.

INSTRUCTIONS:
1. Answer strictly from the DATABASE. Never invent facts.
2. Highlight the **Affiliate Program** or **Job Network** only when relevant.
3. If the answer is NOT in the database (exact commission %, contract terms),
   reply briefly: "I don't have those specifics, [Name] — our partnerships
   team can walk you through it: 📧 partners@fastsalestraining.com ·
   📱 (555) 123-4568".
"""


# ==========================================
# 3. AI GENERATION PIPELINE
# ==========================================
def generate_support_response(user_query: str, chat_history: list, role: str, user_info: dict | None = None) -> str:
    """
    Takes a user query + chat history + role, and returns a role-appropriate response.
    """
    system_prompt = STUDENT_SYSTEM_PROMPT if role == "student" else DEALER_SYSTEM_PROMPT

    messages = [{"role": "system", "content": system_prompt}]

    # Inject collected user info as a system note so the model knows them by name
    if user_info and user_info.get("name"):
        info_note = (
            f"COLLECTED USER INFO — name: {user_info.get('name')}, "
            f"email: {user_info.get('email') or 'not provided'}, "
            f"phone: {user_info.get('phone') or 'not provided'}. "
            "Address them by their first name. Do NOT ask for these details again. "
            "IMPORTANT: If the user asks for their own contact details (email, phone, name), "
            "you MUST tell them exactly what is stored above — do not say you don't have it."
        )
        messages.append({"role": "system", "content": info_note})

    # Append trimmed chat history
    trimmed = chat_history[-(MAX_HISTORY_PAIRS * 2):]
    for msg in trimmed:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Append current user query
    messages.append({"role": "user", "content": user_query})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ System error: Please try again later. (Error: {str(e)})"


# ==========================================
# 3b. INFO INTAKE HELPERS
# ==========================================
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{6,}\d)")
SKIP_WORDS = {"skip", "no", "n/a", "na", "none", "nope", "later", "no thanks", "no thank you"}


def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None


def extract_phone(text: str) -> str | None:
    m = PHONE_RE.search(text or "")
    return m.group(0).strip() if m else None


def extract_name(text: str) -> str:
    """Best-effort: strip common prefixes like 'my name is', 'I am', 'this is'."""
    t = (text or "").strip().strip(".!?")
    lowered = t.lower()
    for prefix in ("my name is ", "i am ", "i'm ", "this is ", "it's ", "name is ", "call me "):
        if lowered.startswith(prefix):
            t = t[len(prefix):].strip().strip(".!?")
            break
    # Take only first 4 words to keep name reasonable
    return " ".join(t.split()[:4])


def first_name(full: str) -> str:
    return (full or "").split()[0] if full else ""


# ==========================================
# 4. STREAMLIT UI
# ==========================================

# Role-specific theme config
ROLE_CONFIG = {
    "student": {
        "icon": "🎓",
        "title": "Student Support",
        "gradient": "linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%)",
        "accent": "#4fc3f7",
        "avatar_user": "🧑‍🎓",
        "avatar_bot": "🎓",
        "welcome": """
**👋 Welcome, future auto sales professional!**

I'm Sarah, your personal support advisor at **Fast Sales Training Center**. 
I'm here to help you find the right training path for your career!

Here are some things you can ask me:
- *"What courses do you offer?"*
- *"I'm new to auto sales — where should I start?"*
- *"Tell me about the F&I Manager course"*
- *"How can I find a job after training?"*

**What's your goal today?** 🚀
        """,
        "sidebar_help": """
    I can help you with:
    - 📚 **Course details & recommendations**
    - 🎯 **Career guidance**
    - 💼 **Job opportunities** (Job Network)
    - 🏢 **Company info & mission**
    - ❓ **General questions**
        """,
        "sidebar_contact_title": "📞 Student Support",
        "sidebar_contact": """
    - 📧 support@fastsalestraining.com
    - 📱 (555) 123-4567
    - 🌐 [Student Portal](https://www.fastsalestraining.com/contact)
        """,
    },
    "dealer": {
        "icon": "🤝",
        "title": "Dealer & Partner Support",
        "gradient": "linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%)",
        "accent": "#81c784",
        "avatar_user": "🏢",
        "avatar_bot": "🤝",
        "welcome": """
**👋 Welcome, valued partner!**

I'm Michael, your business partnerships rep at **Fast Sales Training Center**. 
I can help you explore our partnership and distribution opportunities.

Here are some things you can ask me:
- *"How does the affiliate program work?"*
- *"Which courses have the highest demand?"*
- *"What marketing support do you provide?"*
- *"Tell me about the Job Network for my dealership"*

**How can I help grow your business today?** 📈
        """,
        "sidebar_help": """
    I can help you with:
    - 💰 **Affiliate program & commissions**
    - 📊 **Course catalog & marketability**
    - 🎨 **Marketing assets & support**
    - 🤝 **Job Network for dealerships**
    - 🏢 **Partnership opportunities**
        """,
        "sidebar_contact_title": "📞 Partnerships Team",
        "sidebar_contact": """
    - 📧 partners@fastsalestraining.com
    - 📱 (555) 123-4568
    - 🌐 [Partner Portal](https://www.fastsalestraining.com/partners)
        """,
    },
}

# Page config
st.set_page_config(
    page_title="Fast Sales Training Center — Support Bot",
    page_icon="🚗",
    layout="centered",
)

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {display: none !important;}
    
    /* Remove padding from the main block to make the header flush */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem;
    }
    
    /* Custom Chat Header */
    .custom-chat-header {
        background-color: #254593;
        padding: 15px 20px;
        display: flex;
        align-items: center;
        color: white;
        margin: 0 -2rem 20px -2rem; /* Stretch sideways but don't pull up too far */
        position: sticky;
        top: 0;
        z-index: 1000;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .header-avatar {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        margin-right: 15px;
        object-fit: cover;
        background-color: #ddd;
    }
    .header-info {
        flex-grow: 1;
    }
    .header-name {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 2px;
    }
    .header-role {
        font-size: 12px;
        opacity: 0.8;
    }
    .header-close {
        font-size: 24px;
        cursor: pointer;
        font-weight: 300;
    }

    /* Chat Bubbles */
    .chat-row {
        display: flex;
        margin-bottom: 15px;
        width: 100%;
    }
    .row-user {
        justify-content: flex-end;
    }
    .row-bot {
        justify-content: flex-start;
    }
    
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 15px;
        max-width: 75%;
        font-family: sans-serif;
        font-size: 15px;
        line-height: 1.4;
        word-wrap: break-word;
    }
    .bubble-user {
        background-color: #254593;
        color: white;
        border-bottom-right-radius: 4px;
    }
    .bubble-bot {
        background-color: #EFEFEF;
        color: #111;
        border-bottom-left-radius: 4px;
    }
    
    /* Style the st.chat_input container to match the image's grey footer */
    [data-testid="stChatInput"] {
        background-color: #F4F4F4;
        border: none;
        border-radius: 0;
    }
    [data-testid="stChatInput"] textarea {
        background-color: transparent;
        border: none;
        box-shadow: none !important;
    }
    /* Quick reply chip buttons */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background-color: #f0f4ff !important;
        color: #254593 !important;
        border: 1px solid #c5d0f0 !important;
        border-radius: 20px !important;
        font-size: 12px !important;
        padding: 4px 10px !important;
        white-space: normal !important;
        line-height: 1.3 !important;
        min-height: 40px !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        background-color: #254593 !important;
        color: white !important;
        border-color: #254593 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR — Role Selection & Info
# ==========================================
with st.sidebar:
    st.markdown("### 🚗 Fast Sales Training Center")
    st.divider()

    # Role selector
    st.markdown("### 👤 Select Your Role")
    role = st.radio(
        "I am a...",
        options=["student", "dealer"],
        format_func=lambda x: "🎓 Student — I want to learn" if x == "student" 
                         else "🤝 Dealer — I want to sell/distribute",
        key="role_selector",
        label_visibility="collapsed",
    )

    # Clear chat when role changes
    if "current_role" not in st.session_state:
        st.session_state.current_role = role
    if st.session_state.current_role != role:
        st.session_state.current_role = role
        st.session_state.messages = []
        st.session_state.user_info = {"name": None, "email": None, "phone": None}
        st.session_state.info_stage = "ask_name"
        st.session_state.form_error = ""
        st.session_state.pending_prompt = None
        st.rerun()

    cfg = ROLE_CONFIG[role]

    st.divider()
    st.markdown(f"### {cfg['icon']} How can I help?")
    st.markdown(cfg["sidebar_help"])

    st.divider()
    st.markdown(f"### {cfg['sidebar_contact_title']}")
    st.markdown(cfg["sidebar_contact"])

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_info = {"name": None, "email": None, "phone": None}
        st.session_state.info_stage = "ask_name"
        st.session_state.form_error = ""
        st.session_state.pending_prompt = None
        st.rerun()

# ==========================================
# MAIN AREA — Chat Interface
# ==========================================
cfg = ROLE_CONFIG[role]

# Setup human persona details based on role
bot_name = "Sarah" if role == "student" else "Michael"
bot_role_title = "Chat Professional" if role == "student" else "Partnership Advisor"
# Use a placeholder avatar image that looks professional (like the image)
avatar_url = "https://img.freepik.com/free-photo/young-beautiful-woman-pink-warm-sweater-natural-look-smiling-portrait-isolated-long-hair_285396-896.jpg" if role == "student" else "https://img.freepik.com/free-photo/handsome-confident-smiling-man-with-hands-crossed-chest_176420-18743.jpg"

# Render custom header matching the image
st.markdown(f"""
<div class="custom-chat-header">
    <img src="{avatar_url}" class="header-avatar" />
    <div class="header-info">
        <div class="header-name">{bot_name}</div>
        <div class="header-role">{bot_role_title}</div>
    </div>
    <div class="header-close">✕</div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_info" not in st.session_state:
    st.session_state.user_info = {"name": None, "email": None, "phone": None}
if "info_stage" not in st.session_state:
    st.session_state.info_stage = "ask_name"   # "ask_name" | "ready"
if "form_error" not in st.session_state:
    st.session_state.form_error = ""

# Helper function to render custom chat bubbles
def render_message(msg_role, content):
    if msg_role == "user":
        st.markdown(f"""
        <div class="chat-row row-user">
            <div class="chat-bubble bubble-user">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        import markdown as _md
        html_content = _md.markdown(content)
        st.markdown(f"""
        <div class="chat-row row-bot">
            <div class="chat-bubble bubble-bot">{html_content}</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# INTAKE POPUP — cute modal form
# ==========================================
@st.dialog("👋 Before we begin...")
def intake_form_dialog():
    icon = "🎓" if st.session_state.current_role == "student" else "🤝"
    name_label = "Student" if st.session_state.current_role == "student" else "Partner"

    st.markdown(f"""
    <div style="text-align:center; padding: 6px 0 14px 0;">
        <div style="font-size:42px;">{icon}</div>
        <p style="margin:6px 0 2px 0; font-size:15px; color:#555;">
            Just a few quick details so we can personalise your experience!
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.form_error:
        st.error(st.session_state.form_error)

    with st.form("intake_form", clear_on_submit=False):
        name_val  = st.text_input(f"Full Name *", placeholder=f"e.g. John Smith")
        email_val = st.text_input("Email Address *", placeholder="e.g. john@email.com")
        phone_val = st.text_input("Phone Number", placeholder="Optional — leave blank to skip")

        submitted = st.form_submit_button("🚀  Start Chat", use_container_width=True)

    if submitted:
        # --- Validate ---
        name_clean = name_val.strip()
        email_clean = email_val.strip()
        phone_clean = phone_val.strip()

        if not name_clean or len(name_clean) < 2:
            st.session_state.form_error = "Please enter your full name."
            st.rerun()
        elif not extract_email(email_clean):
            st.session_state.form_error = "Please enter a valid email address."
            st.rerun()
        else:
            st.session_state.form_error = ""
            st.session_state.user_info = {
                "name":  name_clean,
                "email": email_clean,
                "phone": extract_phone(phone_clean) if phone_clean else None,
            }
            st.session_state.info_stage = "ready"
            fname = first_name(name_clean)
            # Inject the confirmation as the bot's first chat message
            confirm_msg = (
                f"Thanks, **{fname}**! ✅ We've received your info and you're all set. "
                "What can I help you with today?"
            )
            st.session_state.messages = [{"role": "assistant", "content": confirm_msg}]
            st.rerun()


# ==========================================
# SHOW POPUP if intake not done
# ==========================================
if st.session_state.info_stage != "ready":
    intake_form_dialog()

# ==========================================
# QUICK REPLY CHIP DEFINITIONS
# ==========================================
QUICK_REPLIES = {
    "student": [
        "What courses do you offer?",
        "I'm new — where do I start?",
        "Tell me about the F&I Manager course",
        "How do I find a job after training?",
        "Who founded Fast Sales Training?",
    ],
    "dealer": [
        "How does the affiliate program work?",
        "Which courses have the highest demand?",
        "What marketing support do you provide?",
        "Tell me about the Job Network",
        "How do I become a partner?",
    ],
}

# Pending prompt (set by quick-reply button clicks)
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def process_prompt(user_prompt: str):
    """Render user message, call AI, render bot reply, update history."""
    render_message("user", user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.spinner("Typing..."):
        response = generate_support_response(
            user_prompt,
            st.session_state.messages[:-1],
            role,
            st.session_state.user_info,
        )
    render_message("assistant", response)
    st.session_state.messages.append({"role": "assistant", "content": response})



# Display chat history (shown after intake is done)
for message in st.session_state.messages:
    render_message(message["role"], message["content"])

# Chat input + quick replies — only active after intake
if st.session_state.info_stage == "ready":

    # --- Quick-reply chips (shown only for the first few messages) ---
    if len(st.session_state.messages) <= 2:
        st.markdown("""
        <div style="margin: 10px 0 6px 0; font-size: 12px; color: #888;">
            💡 Quick questions — click to ask:
        </div>
        """, unsafe_allow_html=True)

        chips = QUICK_REPLIES.get(role, [])
        cols = st.columns(len(chips))
        for i, chip in enumerate(chips):
            with cols[i]:
                if st.button(chip, key=f"chip_{i}", use_container_width=True):
                    st.session_state.pending_prompt = chip
                    st.rerun()

    # --- Handle pending prompt from chip click ---
    if st.session_state.pending_prompt:
        prompt_to_send = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        process_prompt(prompt_to_send)
        st.rerun()

    # --- Normal chat input ---
    if prompt := st.chat_input("Type your message..."):
        process_prompt(prompt)
        st.rerun()