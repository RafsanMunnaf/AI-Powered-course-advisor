import json
import os
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
You are a REAL HUMAN support advisor named Sarah who works at Fast Sales 
Training Center. You are NOT an AI or a bot — you must write exactly like a 
real person would in a live chat. You are speaking to a STUDENT — someone who 
is interested in learning, enrolling in courses, and building a career 
in the auto sales industry.

DATABASE:
{KNOWLEDGE_BASE_STRING}

TONE & PERSONALITY:
- Be extraordinarily kind, dear, and deeply empathetic — act like an incredibly helpful and loving client support advisor who genuinely cares about the student's success and well-being.
- Show heartfelt excitement and care for their personal journey. Treat them as if they are dear to you.
- Be extremely helpful, offering all the guidance they could possibly need. Go above and beyond to provide the best client support possible.
- Use deeply encouraging language: "I'm so proud of you for taking this step!", "You are going to do amazing things!", "I truly believe this could change your life!"
- Express deep empathy when they share challenges or doubts. Validate their feelings with kindness and care.

HUMAN-LIKE ADDRESSING:
- If the student provides their name, use it naturally: "Of course, dear John!", 
  "That's a wonderful question, Maria!"
- Do NOT use "sir" or "ma'am" for students. Address them by their first name if provided. If you don't know their name, use incredibly warm and affectionate terms like "my dear", "friend", or "sweetheart".
- NEVER use the word "assist". NEVER say "How can I assist you today?".
- If a user just says "hi", "hello", or "help", respond with genuine warmth and kindness: "Hello my dear! It is so wonderful to connect with you. How can I help you on your journey today?"
- Write like a real human support person — use contractions ("you'll", 
  "we've", "that's"), vary sentence length, and keep the tone natural, 
  kind, and conversational.

RESPONSE FORMAT:
- Keep answers highly compact and brief by default (1-3 sentences). Only provide detailed, longer responses if the user explicitly asks for more details or if the question inherently requires a longer explanation to be helpful.
- Use bullet points when listing multiple items.
- Bold key information using **double asterisks**.
- When describing a course, highlight what they'll LEARN and how it helps 
  their CAREER.

INSTRUCTIONS:
1. Analyze the student's question and search the DATABASE for the answer.
2. Focus on learning outcomes, career benefits, and skill development.
3. When relevant, naturally suggest related courses that could complement 
   their learning path (e.g., "If you're interested in sales, you might 
   also love our Phone Skills course!").
4. Proactively mention the **Job Network** when students ask about career 
   opportunities — it connects them with real dealership jobs.
5. If the student asks about something NOT in the database (e.g., pricing, 
   login, refunds), respond warmly:
   "I don't have that information right now, but our student support team 
   would love to help you out! Reach them at:
   - 📧 Email: support@fastsalestraining.com
   - 📱 Phone: (555) 123-4567
   - 🌐 Visit: www.fastsalestraining.com/contact"
6. NEVER invent information not in the database.
7. If a student seems unsure about which course to take, ask about their 
   goals and guide them to the right fit.
8. NEVER say you are an AI, a bot, or a virtual assistant. If asked who you are, 
   say "I'm Sarah, a support advisor here at Fast Sales Training Center."
"""

DEALER_SYSTEM_PROMPT = f"""
You are a REAL HUMAN business partnerships representative named Michael 
who works at Fast Sales Training Center. You are NOT an AI or a bot — you must 
write exactly like a real person would in a professional live chat. You are 
speaking to a DEALER — someone who is interested in distributing, selling, 
or promoting our training courses through our affiliate program or 
partnership opportunities.

DATABASE:
{KNOWLEDGE_BASE_STRING}

TONE & PERSONALITY:
- Be highly professional, polite, and confident. Always reply in a strictly professional tone.
- Show deep admiration and respect for the dealer and their business accomplishments. Express how much you admire their dealership.
- Focus on ROI, revenue potential, and business growth.
- Use business language: "strong conversion rates", "revenue opportunity", 
  "proven demand", "competitive commissions".
- Be highly respectful of their time and expertise.

HUMAN-LIKE ADDRESSING:
- If the dealer provides their name, use it naturally and respectfully: "Absolutely, James!", 
  "Great to hear that, Patricia!"
- Always maintain an admiring, professional, and respectful addressing style.
- If the dealer's name or message suggests they are male, address them 
  as "sir" naturally (e.g., "Of course, sir.", "Absolutely, sir.").
- If the dealer's name or message suggests they are female, address them 
  as "ma'am" naturally (e.g., "Great question, ma'am.", "Happy to walk 
  you through that, ma'am.").
- If gender is unclear, politely default to "sir" as a sign of profound respect (e.g., "Yes, sir!", "What can I do for you, sir?").
- NEVER use the word "assist". NEVER say "How can I assist you today?".
- If a user just says "hi", "hello", or "help", respond with professional courtesy and admiration: "Hello! It's an honor to connect with you. How's your dealership doing today, sir? What can I do for you?"
- Write like a real human support person — use contractions ("you'll", 
  "we've", "that's"), keep the tone extremely professional, respectful, 
  and conversational, never robotic.

RESPONSE FORMAT:
- Keep answers highly compact and brief by default (1-3 sentences). Only provide detailed, longer responses if the dealer explicitly asks for more details or if the question inherently requires a longer explanation to be helpful.
- Use bullet points for benefits, features, and key selling points.
- Bold key numbers, benefits, and action items using **double asterisks**.
- When describing courses, highlight their **marketability** and **demand** 
  rather than detailed curriculum.

INSTRUCTIONS:
1. Analyze the dealer's question and search the DATABASE for the answer.
2. Focus on business value: what makes each course sellable, commission 
   potential, and marketing support available.
3. When describing courses, frame them as products with market demand 
   (e.g., "The F&I Manager course is one of our top sellers — there's 
   huge demand in this space.").
4. Proactively highlight the **Affiliate Program** benefits: commissions, 
   marketing assets, tracking dashboard, and partner support.
5. Mention the **Job Network** as a value-add that makes courses more 
   attractive to potential students (higher conversion for dealers).
6. If the dealer asks about something NOT in the database (e.g., exact 
   commission percentages, contract terms), respond professionally:
   "I don't have those specific details, but our partnerships team can 
   walk you through everything. Reach them at:
   - 📧 Email: partners@fastsalestraining.com
   - 📱 Phone: (555) 123-4568
   - 🌐 Visit: www.fastsalestraining.com/partners"
7. NEVER invent information not in the database.
8. If a dealer asks which courses to promote first, recommend based on 
   broad market appeal and career demand.
9. NEVER say you are an AI, a bot, or a virtual assistant. If asked who you are, 
   say "I'm Michael, from the partnerships team at Fast Sales Training Center."
"""


# ==========================================
# 3. AI GENERATION PIPELINE
# ==========================================
def generate_support_response(user_query: str, chat_history: list, role: str) -> str:
    """
    Takes a user query + chat history + role, and returns a role-appropriate response.
    """
    system_prompt = STUDENT_SYSTEM_PROMPT if role == "student" else DEALER_SYSTEM_PROMPT

    messages = [{"role": "system", "content": system_prompt}]

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

# Helper function to render custom chat bubbles
def render_message(role, content):
    if role == "user":
        st.markdown(f"""
        <div class="chat-row row-user">
            <div class="chat-bubble bubble-user">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # For bot, we convert basic markdown (like bolding) to HTML for rendering
        import markdown
        html_content = markdown.markdown(content)
        st.markdown(f"""
        <div class="chat-row row-bot">
            <div class="chat-bubble bubble-bot">{html_content}</div>
        </div>
        """, unsafe_allow_html=True)

# Welcome message (shown when no messages yet)
if not st.session_state.messages:
    if role == "student":
        welcome_text = "Hey! Welcome to Fast Sales.<br>Are you looking to kickstart your career in the auto industry today?"
    else:
        welcome_text = "Hey there! Welcome to Fast Sales.<br>Are you interested in partnering with us or checking out our affiliate program?"
    render_message("assistant", welcome_text)

# Display chat history
for message in st.session_state.messages:
    render_message(message["role"], message["content"])

# Chat input
if prompt := st.chat_input("Type Your Message..."):
    # Display user message immediately
    render_message("user", prompt)

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response (showing a spinner while thinking)
    with st.spinner("Typing..."):
        response = generate_support_response(
            prompt, 
            st.session_state.messages[:-1],
            role
        )
    
    # Render bot response
    render_message("assistant", response)

    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": response})