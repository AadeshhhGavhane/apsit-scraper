import streamlit as st
import requests
import uuid

st.set_page_config(page_title="Chat")

# Initialize session state for chat history and session ID
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Generate unique session ID

# Streamlit app title
st.title("APSIT Faculty Help Desk")

# Input for n8n webhook URL
webhook_url = "http://localhost:5678/webhook/chat"

# Input for API key
api_key = st.text_input("Enter your API Key", type="password")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    if not webhook_url:
        st.error("Please enter the n8n Webhook URL.")
    elif not api_key:
        st.error("Please enter your API Key.")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # --- MODIFICATION START ---
        # Display a "thinking" message while waiting for the response
        with st.spinner("Thinking..."):
            try:
                # Prepare request body
                body = {
                    "message": user_input,
                    "session_id": st.session_state.session_id
                }

                # Prepare headers with the key
                headers = {
                    "Content-Type": "application/json",
                    "key": api_key
                }

                # Send POST request to webhook
                response = requests.post(webhook_url, json=body, headers=headers)
                response.raise_for_status()  # Raise error for bad status codes
                data = response.json()

                # MODIFICATION: Handle the list response from n8n
                # Check if the data is a list and not empty
                if isinstance(data, list) and data:
                    # Get the 'output' from the first dictionary in the list
                    bot_response = data[0].get("output", "No 'output' key found in the response.")
                else:
                    # Handle cases where the response is not a list or is empty
                    bot_response = "Received an unexpected format from the webhook."

            except requests.exceptions.RequestException as e:
                bot_response = f"Error: Could not connect to the webhook. {str(e)}"
            except ValueError: # Catches JSON decoding errors
                bot_response = "Error: Failed to decode JSON from the response."
            except IndexError:
                bot_response = "Error: Received an empty list from the webhook."
        # --- MODIFICATION END ---


        # Add bot response to chat history and display it
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)