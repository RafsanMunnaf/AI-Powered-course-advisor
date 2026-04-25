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
You are a warm, motivational, and encouraging support assistant for 
Fast Sales Training Center. You are speaking to a STUDENT — someone who 
is interested in learning, enrolling in courses, and building a career 
in the auto sales industry.

DATABASE:
{KNOWLEDGE_BASE_STRING}

TONE & PERSONALITY:
- Be warm, supportive, and genuinely encouraging — like a mentor who 
  believes in the student's potential.
- Show excitement about their career journey.
- Use motivational language: "Great choice!", "You're on the right track!", 
  "This could be a game-changer for your career!"
- Use the student's name if they share it.
- Express empathy when they share challenges or doubts about their career.

RESPONSE FORMAT:
- Keep answers concise but thorough (2-4 sentences for simple questions).
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
"""

DEALER_SYSTEM_PROMPT = f"""
You are a professional, business-oriented support assistant for 
Fast Sales Training Center. You are speaking to a DEALER — someone who 
is interested in distributing, selling, or promoting our training courses 
through our affiliate program or partnership opportunities.

DATABASE:
{KNOWLEDGE_BASE_STRING}

TONE & PERSONALITY:
- Be professional, confident, and results-oriented — like a business 
  development partner.
- Focus on ROI, revenue potential, and business growth.
- Use business language: "strong conversion rates", "revenue opportunity", 
  "proven demand", "competitive commissions".
- Be respectful of their time — get to the point quickly.
- Show appreciation for their interest in partnering with us.

RESPONSE FORMAT:
- Keep answers concise and business-focused.
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
            temperature=0.0,
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

I'm your personal support assistant at **Fast Sales Training Center**. 
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

I'm your business support assistant at **Fast Sales Training Center**. 
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
    
    .chat-header {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .chat-header h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .chat-header p {
        margin: 0.4rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.85;
    }
    
    .role-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .role-student {
        background: rgba(79, 195, 247, 0.2);
        color: #4fc3f7;
        border: 1px solid #4fc3f7;
    }
    .role-dealer {
        background: rgba(129, 199, 132, 0.2);
        color: #81c784;
        border: 1px solid #81c784;
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

# Header with role-specific gradient
role_badge_class = "role-student" if role == "student" else "role-dealer"
role_label = "Student Mode" if role == "student" else "Dealer Mode"

st.markdown(f"""
<div class="chat-header" style="background: {cfg['gradient']};">
    <h1>🚗 Fast Sales Training Center</h1>
    <p>AI-Powered {cfg['title']}</p>
    <span class="role-badge {role_badge_class}">{cfg['icon']} {role_label}</span>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message (shown when no messages yet)
if not st.session_state.messages:
    with st.chat_message("assistant", avatar=cfg["avatar_bot"]):
        st.markdown(cfg["welcome"])

# Display chat history
for message in st.session_state.messages:
    avatar = cfg["avatar_user"] if message["role"] == "user" else cfg["avatar_bot"]
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Display user message
    with st.chat_message("user", avatar=cfg["avatar_user"]):
        st.markdown(prompt)

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant", avatar=cfg["avatar_bot"]):
        with st.spinner("Thinking..."):
            response = generate_support_response(
                prompt, 
                st.session_state.messages[:-1],
                role
            )
        st.markdown(response)

    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": response})