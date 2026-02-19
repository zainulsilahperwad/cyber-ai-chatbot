import streamlit as st
import requests
import uuid

# Config & Theme
st.set_page_config(page_title="CyberAI", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stChatMessage { border-radius: 10px; border: 1px solid #30363d; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "detected_lang" not in st.session_state:
    st.session_state.detected_lang = "Waiting..."

# Sidebar
with st.sidebar:
    st.header("🛡️ Session Control")
    st.success(f"**Language:** {st.session_state.detected_lang}")
    if st.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.detected_lang = "Waiting..."
        st.rerun()

st.title("Cyber Security AI")

# 1. Always display history first
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle New Input
if prompt := st.chat_input("Query..."):
    # Display user message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant logic
    with st.chat_message("assistant"):
        # --- THINKING BLOCK ---
        with st.status("CyberAI is thinking...", expanded=True) as status:
            try:
                st.write("Analyzing query for threats...")
                payload = {"message": prompt, "session_id": st.session_state.session_id}
                
                response = requests.post("http://127.0.0.1:8000/chat", json=payload, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("reply")
                    st.session_state.detected_lang = data.get("language")
                    
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                else:
                    status.update(label="Server Error", state="error")
                    answer = "Sorry, I encountered an error with the server."
            except Exception as e:
                status.update(label="Connection Failed", state="error")
                answer = "I couldn't connect to the backend."

        # 3. SHOW THE REPLY (Outside the status block so it stands out)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Rerun to update the sidebar language and ensure sync
        st.rerun()