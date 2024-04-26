import streamlit as st
import os
import requests
#import torch
#import transformers

# Get a response from the LLM API
def get_llm_response(query):
 
    response = requests.post(llm_api_url,
        auth=(
            llm_api_key,
            llm_api_key
        ),
        json={
            "data": {"prompt": query}
        }
    )
    return response.json()['result']

# Change these accordingly. 
# Right now this is refering to the API from the lama_sft_model_sentry project created from the repo at https://github.com/dominodatalab/lama_sft_model_sentry

# specify the model api details
os.environ['domino_model_endpoint']='https://prod-field.cs.domino.tech:443/models/662ba7e8892b9833832b4fad/latest/model'
llm_api_url = os.environ.get('domino_model_endpoint')
llm_api_key = os.environ.get('llm_api_key')

# Set up streamlit and call the LLM API to get an answer to the user's query
st.session_state.setdefault("messages", [])
# Sidebar with Clear Conversation button
with st.sidebar:
    st.title("Settings")
    if st.button('Clear Conversation'):
        st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input and generate response
if (user_input := st.chat_input("How can I help?")):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm_api_response = get_llm_response(user_input)
            answer = llm_api_response['text_from_llm']
        
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
