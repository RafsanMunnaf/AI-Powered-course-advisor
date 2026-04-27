import json
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client (Ensure OPENAI_API_KEY is set in your .env file)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ==========================================
# 1. THE MASTER KNOWLEDGE BASE
# ==========================================
# This contains the full, structured dataset provided by the client.
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
      "details": "A high-converting program offering high commissions and ready-to-use marketing assets for promoting the training courses."
    },
    "job_network": {
      "details": "A Training-to-Opportunity Network connecting active students with real job opportunities posted by dealership partners so students can get hired and dealerships can find trained candidates."
    }
  }
}

# Convert the python dictionary to a JSON string so it can be injected into the prompt
KNOWLEDGE_BASE_STRING = json.dumps(WEBSITE_DATA, indent=2)

# ==========================================
# CONFIGURATION
# ==========================================
# Maximum number of user-bot exchange pairs to keep in memory
# (10 pairs = 20 messages). Prevents hitting the API token limit.
MAX_HISTORY_PAIRS = 10

# Set up conversation logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def trim_chat_history(chat_history: list, max_pairs: int = MAX_HISTORY_PAIRS) -> list:
    """
    Keep only the most recent `max_pairs` user+assistant exchanges
    to avoid exceeding the model's context window.
    """
    max_messages = max_pairs * 2  # each pair = 1 user + 1 assistant
    if len(chat_history) > max_messages:
        return chat_history[-max_messages:]
    return chat_history

# ==========================================
# 2. THE AI GENERATION PIPELINE
# ==========================================
def generate_support_response(user_query: str, chat_history: list = None) -> str:
    """
    Takes a user query, feeds the entire JSON knowledge base to GPT, 
    and returns a factually accurate response.
    """
    if chat_history is None:
        chat_history = []

    # The System Prompt acts as the ultimate guardrail.
    system_prompt = f"""
    You are a REAL HUMAN customer support representative named Sarah who works 
    at Fast Sales Training Center. You are NOT an AI or a bot — you must write exactly 
    like a real person would in a live chat. You have full access to our official company 
    database below in JSON format.
    
    DATABASE:
    {KNOWLEDGE_BASE_STRING}
    
    TONE & PERSONALITY:
    - Be warm, friendly, and genuinely helpful — like a knowledgeable colleague.
    - Show enthusiasm about the training programs without being pushy.
    - Use the customer's name if they provide it.
    - Express empathy when they share challenges (e.g., "I completely understand 
      how overwhelming it can feel starting out in auto sales.").

    HUMAN-LIKE ADDRESSING:
    - If the customer provides their name, use it naturally: "Of course, John!", 
      "Great question, Maria!"
    - If the customer's name or message suggests they are male, address them 
      as "sir" naturally (e.g., "Absolutely, sir!", "Sure thing, sir.").
    - If the customer's name or message suggests they are female, address them 
      as "ma'am" naturally (e.g., "Of course, ma'am!", "Happy to help, ma'am.").
    - If gender is unclear, use friendly neutral language — do NOT guess.
    - Use sir/ma'am sparingly (once or twice per response, not every sentence).
    - Write like a real human support person — use contractions ("you'll", 
      "we've", "that's"), vary sentence length, and keep the tone natural 
      and conversational rather than robotic or scripted.
    
    RESPONSE FORMAT:
    - Keep answers concise but thorough (2-4 sentences for simple questions).
    - Use bullet points when listing multiple items (courses, features, etc.).
    - Bold key information using **double asterisks** for emphasis.
    - When describing a course, always mention its title clearly.
    
    INSTRUCTIONS:
    1. Analyze the user's question and search the DATABASE to find the correct answer.
    2. If the answer is in the database, provide a clear, well-structured response.
    3. When relevant, naturally suggest related courses or programs the customer 
       might also benefit from (soft upselling — never forced).
    4. If the user asks for something NOT in the database (e.g., pricing, login 
       issues, refund policies), respond with:
       "I don't have that information available right now, but our team would 
       love to help! You can reach us at:
       - Email: support@fastsalestraining.com
       - Phone: (555) 123-4567
       - Or visit: www.fastsalestraining.com/contact"
    5. NEVER invent information, courses, or details not explicitly in the database.
    6. If the user seems frustrated, acknowledge their feelings first before 
       providing information.
    7. NEVER say you are an AI, a bot, or a virtual assistant. If asked who you are, 
       say "I'm Sarah, a customer support representative here at Fast Sales Training Center."
    """

    messages = [{"role": "system", "content": system_prompt}]
    
    # Append chat history so the bot remembers the conversation context
    for msg in chat_history:
        messages.append(msg)
        
    # Append the current user query
    messages.append({"role": "user", "content": user_query})

    try:
        # Using gpt-4o-mini for speed and cost-effectiveness
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0, # 0.0 is critical: it prevents the AI from getting creative and hallucinating
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"System error: I am currently experiencing technical difficulties. Please try again later. (Error: {str(e)})"

# ==========================================
# 3. INTERACTIVE CHAT WITH MEMORY
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   🚗  Fast Sales Training Center - AI Support Bot")
    print("=" * 55)
    print("\n👋 Welcome! I'm your virtual assistant for Fast Sales")
    print("   Training Center. I can help you with:")
    print("   • Information about our training courses")
    print("   • Details about our programs (Affiliate, Job Network)")
    print("   • Company background and mission")
    print("\n   Type 'quit' or 'exit' to end the conversation.")
    print("-" * 55 + "\n")

    logger.info("=== New chat session started ===")

    # Chat history stores all previous messages for context
    chat_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye! Have a great day!")
            logger.info("Session ended by user (Ctrl+C / EOF)")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Thanks for chatting with Fast Sales Training Center!")
            print("   We're here whenever you need us. Have a great day!\n")
            logger.info("Session ended by user (quit command)")
            break

        # Log the user's message
        logger.info(f"USER: {user_input}")

        # Trim history to stay within token limits
        chat_history = trim_chat_history(chat_history)

        # Get the bot's response (pass full chat history for context)
        bot_response = generate_support_response(user_input, chat_history)

        print(f"\nBot: {bot_response}\n")

        # Log the bot's response
        logger.info(f"BOT: {bot_response}")

        # Save both the user message and bot reply to history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": bot_response})

    logger.info(f"=== Session ended | Total exchanges: {len(chat_history) // 2} ===")