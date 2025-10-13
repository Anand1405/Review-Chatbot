import base64
import os
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from agent_graph import answer_question, build_graph
from conversation_store import ConversationStore

st.set_page_config(page_title="Liquide Review Insights Agent", page_icon="📊", layout="wide")


def init_page_state():
    """Initialise session state variables on first page load."""
    if "graph" not in st.session_state:
        try:
            db_path = st.secrets.get("db_path", None)  # type: ignore[assignment]
        except StreamlitSecretNotFoundError:
            db_path = None
        if not db_path:
            db_path = os.environ.get("REVIEWS_DB_PATH", "reviews.db")
        st.session_state.graph = build_graph(db_path)
    if "conversation_store" not in st.session_state:
        conversations_db = os.environ.get("CONVERSATIONS_DB_PATH", "conversations.db")
        st.session_state.conversation_store = ConversationStore(conversations_db)
    if "thread_id" not in st.session_state:
        store = st.session_state.conversation_store
        threads = store.list_threads()
        if threads:
            st.session_state.thread_id = threads[0]["id"]
        else:
            st.session_state.thread_id = store.create_thread()


def apply_theme():
    """Inject custom CSS for background image and colors."""
    primary = "#1b2748"
    secondary = "#253a60"
    accent = "#3d5a94"
    text = "#f2f4ff"

    background_css = (
        "background: linear-gradient(135deg, rgba(27,39,72,0.95) 0%, rgba(32,48,86,0.92) 45%, rgba(21,27,41,0.96) 100%);"
    )

    background_path = Path("background.jpg")
    if background_path.exists():
        try:
            encoded = base64.b64encode(background_path.read_bytes()).decode()
            background_css = (
                f'background-image: url("data:image/png;base64,{encoded}");'
                "background-size: cover;"
                "background-position: center;"
                "background-attachment: fixed;"
                "background-repeat: no-repeat;"
            )
        except Exception:
            pass

    st.markdown(
        f"""
        <style>
        .stApp {{
            {background_css}
            color: {text};
        }}
        .stSidebar {{
            background: rgba(20, 30, 50, 0.85);
            color: {text};
        }}
        .stButton > button {{
            background-color: {accent};
            color: {text};
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.15);
        }}
        .stButton > button:hover {{
            background-color: {secondary};
            border-color: rgba(255,255,255,0.35);
        }}
        div[data-baseweb="select"] > div {{
            background-color: rgba(255,255,255,0.05) !important;
            color: {text} !important;
        }}
        div.stChatMessage {{
            border-radius: 12px;
            padding: 0.8rem;
            margin-bottom: 0.6rem;
            backdrop-filter: blur(4px);
        }}
        div.stChatMessage.user {{
            background-color: rgba(255,255,255,0.08);
            color: {text};
        }}
        div.stChatMessage.assistant {{
            background-color: rgba(48, 72, 120, 0.55);
            color: {text};
        }}
        .stTextInput > div > div > input {{
            background-color: rgba(255,255,255,0.08);
            color: {text};
        }}
        .stTextArea textarea {{
            background-color: rgba(255,255,255,0.08);
            color: {text};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_chat():
    """Render the chat interface and handle user input."""
    apply_theme()
    store: ConversationStore = st.session_state.conversation_store
    thread_id: str = st.session_state.thread_id

    st.title("Liquide Review Insights Agent")
    st.write(
        "Ask questions about Liquide app reviews. The agent will return summaries, stats,"
        " evidence, applied filters and the SQL used."
    )

    threads = store.list_threads()
    if not threads:
        thread_id = store.create_thread()
        st.session_state.thread_id = thread_id
        threads = store.list_threads()

    def _thread_label(thread: dict, idx: int) -> str:
        title = thread["title"].strip()
        if title:
            return title
        return f"Conversation {idx + 1}"

    thread_options = {thread["id"]: _thread_label(thread, idx) for idx, thread in enumerate(threads)}
    default_index = next((idx for idx, thread in enumerate(threads) if thread["id"] == thread_id), 0)

    with st.sidebar:
        st.subheader("Conversations")
        selected = st.selectbox(
            "Choose a conversation",
            list(thread_options.keys()),
            index=default_index,
            format_func=lambda key: thread_options[key],
        )
        if st.button("New conversation"):
            new_id = store.create_thread()
            st.session_state.thread_id = new_id
            st.rerun()

        if selected != thread_id:
            st.session_state.thread_id = selected
            st.rerun()

    thread_id = st.session_state.thread_id

    for message in store.get_messages(thread_id):
        role = message["role"]
        display_role = "assistant" if role == "assistant" else "user"
        st.chat_message(display_role).markdown(
            message["content"],
            unsafe_allow_html=role == "assistant",
        )

    if question := st.chat_input("Enter your question"):
        with st.spinner("Generating answer..."):
            answer_markdown = answer_question(
                st.session_state.graph,
                question,
                thread_id=thread_id,
                store=store,
            )
        st.rerun()


if __name__ == "__main__":
    init_page_state()
    run_chat()
