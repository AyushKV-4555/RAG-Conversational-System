import streamlit as st
from backend_rag_tool import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# ========================= Page Config =========================
st.set_page_config(
    page_title="Multi Utility Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ========================= Helpers =========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# ========================= Session Init =========================
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

thread_id = st.session_state["thread_id"]
thread_docs = st.session_state["ingested_docs"].setdefault(thread_id, {})

# ========================= Sidebar =========================
st.sidebar.markdown("## 🤖 LangGraph Chatbot")

st.sidebar.markdown(
    f"""
**🧵 Active Chat**
Thread ID: `{thread_id}`
"""
)

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# -------- PDF Section --------
st.sidebar.markdown("### 📄 Document Context")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF (optional)",
    type=["pdf"]
)

if uploaded_pdf:
    if uploaded_pdf.name not in thread_docs:
        with st.sidebar.status("Indexing document…", expanded=True):
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_id,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            st.sidebar.success("PDF indexed successfully!")
    else:
        st.sidebar.info("This PDF is already indexed.")

if thread_docs:
    latest = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"📘 **Using:** {latest['filename']}\n\n"
        f"📄 Pages: {latest['documents']}  \n"
        f"🧩 Chunks: {latest['chunks']}"
    )
else:
    st.sidebar.info("No document uploaded for this chat.")

st.sidebar.divider()

# -------- Past Chats --------
st.sidebar.markdown("### 🕘 Past Chats")
for tid in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(tid[:12] + "...", key=tid):
        st.session_state["thread_id"] = tid
        msgs = load_conversation(tid)
        st.session_state["message_history"] = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in msgs
        ]
        st.rerun()

# ========================= Main Chat UI =========================
st.title("💬 Multi Utility Chatbot")

for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything or query your document…")

# ========================= Chat Logic =========================
if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant", avatar="🤖"):
        status_holder = {"box": None}

        def stream_response():
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using tool: `{chunk.name}`", expanded=False
                        )

                if isinstance(chunk, AIMessage):
                    yield chunk.content

        assistant_reply = st.write_stream(stream_response())

        if status_holder["box"]:
            status_holder["box"].update(
                label="✅ Tool completed",
                state="complete"
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": assistant_reply}
    )
