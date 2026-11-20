"""Streamlit chat UI for the Acme assistant.

Run:  .\\.venv\\Scripts\\streamlit.exe run app.py
Then open http://localhost:8501 in your browser.
"""

import streamlit as st

import agent
import memory
from rag.permissions import ROLES

st.set_page_config(page_title="Acme Assistant", page_icon="🧭")

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history shown on screen

# --- Sidebar: who am I + what do you remember about me -----------------------
with st.sidebar:
    st.header("👤 You")
    user_id = st.text_input("User ID", value=agent.DEFAULT_USER,
                            help="Memories are stored per user ID.").strip() or agent.DEFAULT_USER
    role = st.selectbox("Role", ROLES, help="Decides which company documents you may see.")

    st.header("🧠 Memory")
    memories = memory.list_memories(user_id)
    if memories:
        for m in memories:
            st.markdown(f"- {m['text']}")
    else:
        st.caption("Nothing remembered yet. Try: “Please remember that I am vegetarian.”")

    # Destructive action -> a human must confirm (the LLM can never do this)
    sure = st.checkbox("Yes, delete everything you remember about me")
    if st.button("🗑️ Forget me", disabled=not sure or not memories):
        deleted = memory.forget_me(user_id)["deleted"]
        st.success(f"Deleted {deleted} memories.")
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def show_details(msg):
    """The 'Tools used' panel under an assistant answer."""
    for note in msg.get("freshness", []):
        st.caption(f"🕒 {note}")
    if msg.get("memories_used"):
        st.caption("🧠 Used memories: " + "; ".join(msg["memories_used"]))
    trace = msg.get("trace", [])
    with st.expander(f"🔧 Tools used ({len(trace)})"):
        if not trace:
            st.write("No tools were needed.")
        for t in trace:
            args = {k: v for k, v in t["arguments"].items() if k not in ("role", "user_id")}
            st.markdown(f"**Step {t['step']} · `{t['tool']}`** `{args}`")
            st.write("→ " + agent.short(t["result"]))
            for chunk in t["result"].get("results", []):  # document passages + sources
                st.caption(f"📄 {chunk['source']}: {chunk['text'][:200]}…")


# --- Main: the chat ----------------------------------------------------------
st.title("🧭 Acme Assistant")
st.caption("Company policies · live currency · live weather · long-term memory")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_details(msg)

if question := st.chat_input("Ask about policies, money, weather… or tell me something to remember"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = agent.run_agent(question, role, user_id=user_id)
        reply = {"role": "assistant", "content": result["answer"], "trace": result["trace"],
                 "freshness": result.get("freshness", []),
                 "memories_used": result.get("memories_used", [])}
        st.markdown(reply["content"])
        show_details(reply)
    st.session_state.messages.append(reply)
    if any(t["tool"] == "save_memory" for t in result["trace"]):
        st.rerun()  # refresh the memory sidebar
