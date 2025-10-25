import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000/api/v1/chat/chat"  
SESSION_ID = "streamlit_session_001"

st.set_page_config(page_title="Neuraline Chat", page_icon="🧠", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "sender": "neuraline",
        "text": (
            "Hey there 👋\n\n"
            "I'm Neuraline, your AI companion for focus, emotional clarity, and personal growth.\n"
            "How are you feeling today — calm, scattered, or somewhere in between?"
        )
    })

st.title("🧠 Neuraline Chat")
st.caption("Welcome to Neuraline — a space to reflect, refocus, and reconnect with your inner clarity.")


chat_container = st.container()

for msg in st.session_state.messages:
    if msg["sender"] == "user":
        chat_container.chat_message("user").write(msg["text"])
    else:
        chat_container.chat_message("assistant").write(msg["text"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"sender": "user", "text": user_input})
    chat_container.chat_message("user").write(user_input)

    with chat_container.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("🤔 Neuraline is thinking...")

    payload = {"message": user_input, "session_id": SESSION_ID}

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        data = response.json()

        reply = data.get("reply")
        best_role = data.get("best_role", "")
        if not reply:
            reply = (
                "I'm here with you. It seems I couldn't reach full context this time — "
                "could you rephrase that, or tell me a bit more?"
            )

        if best_role:
            reply += f"\n\n*(role: {best_role})*"

        time.sleep(0.5)
        placeholder.write(reply)

        st.session_state.messages.append({"sender": "neuraline", "text": reply})

    except requests.exceptions.RequestException as e:
        error_msg = f"⚠️ Could not connect to backend: {e}"
        placeholder.write(error_msg)
        st.session_state.messages.append({"sender": "neuraline", "text": error_msg})

st.sidebar.header("🧩 Session Controls")
if st.sidebar.button("Clear chat history"):
    st.session_state.messages = []
    st.experimental_rerun()

st.sidebar.info("Neuraline Streamlit UI — warm, conversational, and functional.")