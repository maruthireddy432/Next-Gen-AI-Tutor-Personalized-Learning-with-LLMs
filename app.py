import streamlit as st
import re
import sys
from io import StringIO
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory

# --- 1. CONFIGURATION & STATE ---
st.set_page_config(page_title="AI Persona Tutor", layout="wide")

if "messages" not in st.session_state: st.session_state.messages = []
if "memory" not in st.session_state: st.session_state.memory = ConversationBufferMemory(memory_key="history", input_key="input")
if "quiz_data" not in st.session_state: st.session_state.quiz_data = None

# --- 2. SIDEBAR NAVIGATION & PROFILE ---
with st.sidebar:
    st.title("🚀 Navigation")
    page = st.radio("Go to:", ["🏠 Home", "🎓 Tutor Chat", "📝 Quiz Room"])
    
    st.divider()
    st.header("🔑 Settings")
    api_key = st.text_input("Groq API Key", type="password")
    # model = st.selectbox("Model", ["openai/gpt-oss-120b", "mixtral-8x7b-32768"])
    model="openai/gpt-oss-120b"
    
    st.header("👤 Profile")
    subject = st.selectbox("Subject", ["Python", "Machine Learning", "Deep Learning"])
    level = st.select_slider("Level", ["Beginner", "Intermediate", "Advanced"])
    style = st.radio("Style", ["Code-first", "Mathematical", "Conceptual"])
    
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

# --- 3. HELPER FUNCTIONS ---
def get_response(prompt_text, use_memory=True):
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        return None
    llm = ChatGroq(model=model, groq_api_key=api_key, temperature=0.3)
    if use_memory:
        template = PromptTemplate.from_template("Tutor in {subject} for {level}. Style: {style}. History: {history}. Input: {input}")
        chain = LLMChain(llm=llm, prompt=template, memory=st.session_state.memory)
        # Ensure we return .content string
        res = chain.invoke({"subject": subject, "level": level, "style": style, "input": prompt_text})
        return res["text"] if isinstance(res, dict) else res.content
    else:
        res = llm.invoke(prompt_text)
        return res.content

def render_sandbox(text, idx):
    blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
    for i, code in enumerate(blocks):
        with st.expander(f"Interactive Code {i+1}"):
            edited = st.text_area("Edit Code:", value=code, height=150, key=f"ed_{idx}_{i}")
            if st.button("Run", key=f"btn_{idx}_{i}"):
                buffer = StringIO()
                sys.stdout = buffer
                try: exec(edited); st.code(buffer.getvalue())
                except Exception as e: st.error(e)
                finally: sys.stdout = sys.__stdout__

# --- 4. PAGE LOGIC ---

# PAGE: HOME
if page == "🏠 Home":
    st.title("👋 Welcome to your AI Personalized Tutor")
    st.markdown(f"""
    Hello **learner**! This application is your dedicated space for mastering your Learning.Happy Learning!
    
    ### 🌟 Core Features
    - **🎓 Tutor Chat:** Engage in a deep technical conversation. The AI adapts to your skill level and preferred style.
    - **📝 Quiz Room:** Test your knowledge! Generate custom quizzes and get them evaluated instantly by the AI.
    - **🛠️ Code Sandbox:** Run and edit Python code snippets directly within the chat.
    
    **How to start?**
    1. Enter your **Groq API Key** in the sidebar.
    2. Choose a page from the navigation menu above to begin your journey.
    """)
    

# PAGE: TUTOR CHAT
elif page == "🎓 Tutor Chat":
    st.title("🎓 Adaptive Tutor Chat")
    for idx, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant": render_sandbox(m["content"], idx)

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = get_response(prompt)
            if response:
                st.markdown(response)
                render_sandbox(response, len(st.session_state.messages))
                st.session_state.messages.append({"role": "assistant", "content": response})

# PAGE: QUIZ ROOM
elif page == "📝 Quiz Room":
    st.title("📝 Technical Quiz Module")
    topic = st.text_input("Enter topic for your quiz:")
    
    if st.button("Generate Quiz") and topic:
        with st.spinner("Creating your quiz..."):
            quiz_prompt = f"Generate a new set of random 5-questions quiz on {topic} for a {level} learner in {subject}. Provide questions and options only."
            st.session_state.quiz_data = get_response(quiz_prompt, use_memory=False)
            st.session_state.last_topic = topic

    if st.session_state.quiz_data:
        st.info(st.session_state.quiz_data)
        ans = st.text_area("Write your answers/code here:")
        if st.button("Submit Answers"):
            with st.spinner("Evaluating..."):
                eval_prompt = f"""Topic: {st.session_state.last_topic}. Quiz: {st.session_state.quiz_data}. User Answers: {ans}. Evaluate technical accuracy and provide a score out of 10. 
                If user gives wrong answer provide them explanation with the correct answer and if score is below 6, provide encouragement and tell them to learn again and come back later when they're ready."""
                evaluation = get_response(eval_prompt, use_memory=False)
                st.success("### Evaluation Results")
                st.markdown(evaluation)