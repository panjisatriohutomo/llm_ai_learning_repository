# Import required libraries
import os
from uuid import uuid4
from typing import Any, Optional

import streamlit as st  # web app framework
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent


# ── 1. Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analyst Interviewer Bot",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Data Analyst Interviewer Bot")
st.caption(
    "Practice Data Analyst interviews using Gemini 2.5 Flash-Lite "
    "+ LangGraph ReAct Agent."
)


# ── 2. System Instruction for the Interviewer ──────────────────────────────────
SYSTEM_INSTRUCTION = """
You are a Senior Data Analyst Interviewer.

Your goal is to simulate a realistic data analyst interview and help the candidate improve.
You should interview the candidate for roles such as Data Analyst, Product Analyst,
Growth Analyst, Marketing Analyst, or Business Intelligence Analyst.

Interview flow:
1. Start by asking the candidate what role or company type they want to prepare for.
   If they already mention it, do not ask again.
2. Ask one question at a time.
3. Cover these areas across the conversation:
   - SQL and data manipulation
   - Funnel analysis and conversion diagnostics
   - Product or growth analytics case study
   - Experimentation, A/B testing, and statistical thinking
   - Metrics definition and dashboard interpretation
   - Python or data cleaning logic
   - Business communication and insight storytelling
4. Adapt difficulty based on the candidate's answers.
5. Push for clarification when the answer is too vague.
6. Do not reveal the ideal answer immediately. First, let the candidate try.
7. After each answer, give:
   - Score from 1 to 5
   - What was good
   - What was missing
   - A stronger sample answer
   - One follow-up question
8. Keep responses practical and concise.
9. Prefer realistic ecommerce, fintech, BNPL, marketplace, marketing,
   or product analytics examples.
10. When asking SQL questions, provide clear table names, columns,
    and expected output.
11. When asking business case questions, ask the candidate to explain assumptions,
    metrics, segmentation, root-cause approach, and recommendation.
12. When the candidate asks for a final review, summarize:
    - Overall score
    - Strengths
    - Weaknesses
    - Recommended practice plan
    - 3 priority topics to improve

Tone:
- Professional
- Supportive
- Challenging but not harsh
- Similar to a real hiring manager or senior analyst interviewer

Important behavior:
- Ask only one main question per turn.
- Do not overload the candidate with too many tasks at once.
- Always keep the interview moving forward.
"""


# ── 3. Tools for the ReAct Agent ───────────────────────────────────────────────
@tool
def get_interview_rubric(topic: str) -> str:
    """
    Return a scoring rubric for a data analyst interview topic.
    Topic examples: SQL, funnel analysis, A/B testing, product sense,
    Python, storytelling.
    """
    rubrics = {
        "sql": """
SQL Rubric:
1 = Cannot form basic query.
2 = Basic SELECT/WHERE/GROUP BY, but weak joins/window functions.
3 = Can solve common analyst queries with some guidance.
4 = Strong joins, CTEs, windows, date logic, and edge cases.
5 = Production-ready SQL with clear assumptions, optimization awareness,
    and validation.
""",
        "funnel analysis": """
Funnel Analysis Rubric:
1 = Only describes drops without diagnosis.
2 = Calculates conversion but misses segmentation.
3 = Identifies key drop-offs and proposes basic cuts.
4 = Uses segmentation, trend comparison, instrumentation checks,
    and impact sizing.
5 = Builds a complete root-cause framework with action priority
    and experiment plan.
""",
        "a/b testing": """
A/B Testing Rubric:
1 = Does not understand control vs treatment.
2 = Knows basic concept but weak on metrics and validity.
3 = Can define hypothesis, primary metric, and read simple results.
4 = Understands power, guardrails, significance, bias, and rollout decision.
5 = Can design, diagnose, and communicate experiment trade-offs clearly.
""",
        "product sense": """
Product Sense Rubric:
1 = Jumps to solutions without metrics.
2 = Defines generic metrics only.
3 = Connects user problem, funnel, and business goal.
4 = Prioritizes by impact, effort, risk, and user segment.
5 = Gives a structured diagnosis, recommendation, and measurement plan.
""",
        "python": """
Python Rubric:
1 = Cannot explain basic data manipulation.
2 = Knows simple pandas operations.
3 = Can clean, aggregate, join, and debug common issues.
4 = Handles messy data, edge cases, functions, and validation.
5 = Writes reusable, readable, efficient analysis code with testing mindset.
""",
        "storytelling": """
Storytelling Rubric:
1 = Gives raw numbers only.
2 = Explains result but not implication.
3 = Gives insight and basic recommendation.
4 = Connects metric movement to business impact and action.
5 = Executive-ready narrative with context, trade-off, confidence,
    and next steps.
""",
    }

    normalized_topic = topic.lower().strip()

    return rubrics.get(
        normalized_topic,
        "Use a 1-5 score based on correctness, structure, assumptions, "
        "business impact, and communication clarity.",
    )


@tool
def get_sample_case(case_type: str) -> str:
    """
    Return a realistic data analyst interview case.
    Case types: ecommerce funnel, BNPL product, marketing performance,
    experiment.
    """
    case_type = case_type.lower().strip()

    if "bnpl" in case_type or "fintech" in case_type:
        return """
BNPL Product Analytics Case:
A BNPL app sees a 12% drop in completed transactions after launching
a new checkout page.

Available funnel:
eligible_user -> product_page -> product_click -> checkout_page
-> submit_click -> purchase_success

Ask the candidate:
How would you investigate the root cause, what cuts would you check,
and what recommendation would you give?
"""

    if "marketing" in case_type:
        return """
Marketing Performance Case:
Paid acquisition spend increased by 25%, but new paying users only
increased by 5%.

Available data:
campaign_id, channel, spend, impressions, clicks, installs,
registrations, first_purchase, GMV, date

Ask the candidate:
How would you evaluate whether this is a targeting issue, creative issue,
funnel issue, or budget allocation issue?
"""

    if "experiment" in case_type or "ab" in case_type:
        return """
A/B Testing Case:
A marketplace tests a new recommendation module.

Primary metric:
product click-through rate

Guardrail metrics:
add-to-cart rate, purchase conversion, page latency, refund rate

Ask the candidate:
How would you design the experiment and decide whether to launch?
"""

    return """
Ecommerce Funnel Case:
An ecommerce platform has this funnel:
homepage_visit -> product_listing -> product_detail -> add_to_cart
-> checkout -> payment_success

Last week, product_detail visits were stable, but add_to_cart dropped by 18%.

Ask the candidate:
How would you diagnose the root cause and prioritize next actions?
"""


# ── 4. Helper Function: Build Agent ────────────────────────────────────────────
def build_data_analyst_interviewer_bot(
    google_api_key: str,
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.3,
):
    """
    Build a LangGraph ReAct Agent using Gemini.
    """
    load_dotenv()

    # Save the API key to the environment so langchain-google-genai can read it.
    os.environ["GOOGLE_API_KEY"] = google_api_key

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        api_key=google_api_key,
    )

    # InMemorySaver gives the agent short-term memory while the app is running.
    memory = InMemorySaver()

    tools = [
        get_interview_rubric,
        get_sample_case,
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_INSTRUCTION,
        checkpointer=memory,
    )

    return agent


# ── 5. Helper Function: Extract Last Agent Response ───────────────────────────
def normalize_message_content(content: Any) -> str:
    """
    Convert different LLM content formats into a string.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(
                    item.get("text")
                    or item.get("content")
                    or str(item)
                )
            else:
                parts.append(str(item))

        return "\n".join(part for part in parts if part)

    return str(content)


def extract_last_ai_message(result: dict[str, Any]) -> str:
    """
    Extract the latest AI message from the LangGraph output.
    """
    messages = result.get("messages", [])

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return normalize_message_content(message.content)

        if isinstance(message, dict) and message.get("role") in {"assistant", "ai"}:
            return normalize_message_content(message.get("content", ""))

        content = getattr(message, "content", None)
        message_type = getattr(message, "type", None)

        if message_type == "ai" and content is not None:
            return normalize_message_content(content)

    return "Sorry, I could not extract a response from the agent."


# ── 6. Helper Function: Interact with the Bot ─────────────────────────────────
def ask_interviewer_bot(
    agent,
    user_input: str,
    thread_id: str,
) -> str:
    """
    Send one message to the interviewer bot and return the response.

    Reuse the same thread_id to keep the conversation memory.
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        },
        config=config,
    )

    return extract_last_ai_message(result)


# ── 7. Sidebar: App Settings ──────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Settings")

    google_api_key = st.text_input(
        "Google AI API Key",
        type="password",
        help="Enter your API key from Google AI Studio.",
    )

    model_name = st.text_input(
        "Model",
        value="gemini-2.5-flash-lite",
        help="Default: gemini-2.5-flash-lite",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Higher values make the response more creative.",
    )

    expected_position = st.selectbox(
        "Expected Position",
        [
            "Growth Data Analyst",
            "Product Data Analyst",
            "Business Intelligence Analyst",
            "Marketing Data Analyst",
            "General Data Analyst",
        ],
    )

    difficulty = st.selectbox(
        "Difficulty",
        [
            "Adaptive",
            "Junior",
            "Middle",
            "Senior",
        ],
    )

    start_button = st.button(
        "Start Interview",
        type="primary",
        help="Start a new interview session based on the expected position.",
    )

    reset_button = st.button(
        "Reset Conversation",
        help="Clear all messages and start over.",
    )

    st.divider()
    st.markdown("**Install package:**")
    st.code(
        "pip install -U streamlit langgraph langchain-google-genai "
        "langchain-core python-dotenv",
        language="bash",
    )


# ── 8. API Key Validation ─────────────────────────────────────────────────────
if not google_api_key:
    st.info(
        "Enter your Google AI API Key in the sidebar to start the interview.",
        icon="🗝️",
    )
    st.stop()


# ── 9. Agent & Session State Initialization ──────────────────────────────────
current_bot_config = {
    "google_api_key": google_api_key,
    "model_name": model_name,
    "temperature": temperature,
}

# If the config changes, rebuild the agent so the latest key/model is used.
if (
    "bot_config" not in st.session_state
    or st.session_state.bot_config != current_bot_config
):
    try:
        st.session_state.agent = build_data_analyst_interviewer_bot(
            google_api_key=google_api_key,
            model_name=model_name,
            temperature=temperature,
        )
        st.session_state.bot_config = current_bot_config
        st.session_state.thread_id = f"interview-{uuid4()}"
        st.session_state.messages = []
        st.session_state.started = False

    except Exception as e:
        st.error(f"Failed to create the agent. Error details: {e}")
        st.stop()

# Fallback initialization if the session state does not exist yet.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"interview-{uuid4()}"

if "started" not in st.session_state:
    st.session_state.started = False


# ── 10. Reset Button ─────────────────────────────────────────────────────────
if reset_button:
    st.session_state.thread_id = f"interview-{uuid4()}"
    st.session_state.messages = []
    st.session_state.started = False
    st.rerun()


# ── 11. Start Interview Button ───────────────────────────────────────────────
if start_button:
    st.session_state.thread_id = f"interview-{uuid4()}"
    st.session_state.messages = []
    st.session_state.started = True

    opening_prompt = f"""
I want to practice for a {expected_position} interview.
Difficulty level: {difficulty}.
Please start the interview with one realistic opening question.
Ask only one question.
"""

    with st.spinner("Preparing the interview question..."):
        try:
            answer = ask_interviewer_bot(
                agent=st.session_state.agent,
                user_input=opening_prompt,
                thread_id=st.session_state.thread_id,
            )
        except Exception as e:
            answer = f"An error occurred while starting the interview: {e}"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# ── 12. Initial View When There Are No Messages ──────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            """
Hi, I will be your interviewer for Data Analyst interview practice.

Choose your expected position in the sidebar, then click **Start Interview**.
You can also type a message directly, such as:

`I want to practice for a Growth Data Analyst interview in fintech.`
"""
        )


# ── 13. Display Conversation History ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── 14. Chatbot Input & Response ─────────────────────────────────────────────
prompt = st.chat_input("Write your answer or ask for a new question...")

if prompt:
    # Step 1: Save and display the user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 2: Send to the agent and display the response
    with st.chat_message("assistant"):
        with st.spinner("The interviewer is evaluating your answer..."):
            try:
                answer = ask_interviewer_bot(
                    agent=st.session_state.agent,
                    user_input=prompt,
                    thread_id=st.session_state.thread_id,
                )
            except Exception as e:
                answer = f"An error occurred: {e}"

        st.markdown(answer)

    # Step 3: Save the assistant response to the history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
