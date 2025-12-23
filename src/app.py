import streamlit as st
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="RAG Playground",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Mock State Management (The "Placeholder" Logic) ---
# We initialize variables here to simulate database/session persistence
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "chat_history" not in st.session_state:
    # Mock data to simulate saved history files
    st.session_state.chat_history = [
        {"id": "1", "title": "Contract_v1.pdf", "date": "2023-10-25"},
        {"id": "2", "title": "Arabic_Report.pdf", "date": "2023-10-24"},
    ]

# --- 3. Sidebar (Navigation & Settings) ---
with st.sidebar:
    st.header("🗂️ Chat Sessions")
    
    # 3.1 New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        # Logic: Reset current session
        st.session_state.messages = []
        st.session_state.current_file = None
        st.rerun()

    # 3.2 History List (Mock)
    st.markdown("---")
    st.write("Recent Chats:")
    for chat in st.session_state.chat_history:
        # In real app, clicking this would load the specific JSON file
        if st.button(f"📄 {chat['title']}", key=chat['id']):
            st.toast(f"Loading chat {chat['id']}... (Mock)")

    # 3.3 Settings (LLM Provider)
    st.markdown("---")
    st.subheader("⚙️ Settings")
    llm_choice = st.selectbox(
        "Select LLM Provider",
        ["OpenAI (GPT-4o)", "OpenAI (GPT-3.5)", "Local (Ollama/Llama3)"]
    )
    st.caption(f"Current Model: {llm_choice}")

# --- 4. Main Content Area ---
st.title("🤖 Multilingual RAG Assistant")

# 4.1 File Upload Section (Only show if no file is active)
if not st.session_state.current_file:
    with st.container():
        st.info("Please upload a document to start chatting.")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
        
        if uploaded_file:
            # --- MOCK INGESTION ---
            with st.spinner("Processing document (Chunking & Indexing)..."):
                time.sleep(1.5) # Simulate processing time
                st.session_state.current_file = uploaded_file.name
                st.success("File Processed!")
                st.rerun()

# 4.2 Chat Interface (Only show if file is active)
else:
    # Header showing active file
    st.write(f"📂 **Active Document:** `{st.session_state.current_file}`")
    st.divider()

    # Display Message History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Ask a question in English or Arabic..."):
        # 1. Display User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Simulate AI Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # --- MOCK GENERATION ---
            # In real app, this is where we call rag_engine.get_answer()
            mock_response = f"This is a simulated answer for '{prompt}' using {llm_choice}. \n\n**Source:** Page 3 of {st.session_state.current_file}"
            
            # Simulate streaming
            for chunk in mock_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})