# Fast Sales Training Center - AI Support System

An AI-powered support chatbot designed for the **Fast Sales Training Center**. This system provides a dynamic, role-based conversational experience tailored for two primary user groups: **Students** and **Dealers**. 

Built with Python, OpenAI's GPT-4o-mini, and Streamlit, it offers both a command-line interface and a rich web-based UI.

## 🌟 Key Features

* **Role-Based Personalities (Student vs. Dealer):**
  * **🎓 Student Mode (Ava):** Warm, mentor-like, encouraging, and focused on learning outcomes, career paths, and the Job Network.
  * **🤝 Dealer Mode (Ava):** Professional, business-oriented, focused on ROI, course marketability, affiliate commissions, and marketing support.
* **Intelligent Conversation Memory:** Maintains context across the session (token-safe with a limit on history pairs).
* **Rich Streamlit Web UI:** Features custom styling, role-specific themes (gradients, avatars), quick-action buttons, and sidebar navigation.
* **Knowledge Base Integration:** Answers are grounded in a structured JSON database, preventing hallucinations.
* **Fallback Escalation:** gracefully hands over to human support with appropriate contact info when the AI doesn't know the answer.
* **Conversation Logging:** Automatically logs all chat sessions for quality assurance (saved in `chat_logs/`).

## 📁 File Structure

```text
.
├── app.py               # Streamlit web application (Rich UI, Role selection)
├── main.py              # CLI version of the chatbot (Terminal-based)
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API keys)
├── .gitignore           # Git ignore rules
└── chat_logs/           # Auto-generated conversation logs
```

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RafsanMunnaf/AI-Power-student-and-Course-dealer-support-system.git
   cd AI-Power-student-and-Course-dealer-support-system
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # source venv/bin/activate    # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 💻 Usage

### Run the Web UI (Streamlit)
For the full role-based experience with the graphical interface:
```bash
streamlit run app.py
```
This will open the app in your default web browser (usually at `http://localhost:8501`).

### Run the Command Line Interface
For a simple terminal-based chat session:
```bash
python main.py
```
