import streamlit as st
import replicate
import os
from transformers import AutoTokenizer

# Set assistant icon to a pen logo
icons = {"assistant": "./Snowflake_Logomark_blue.svg", "user": "✏️"}
# App title
st.set_page_config(page_title="Content Writer Assistant")

# Replicate Credentials
with st.sidebar:
    st.title('Content Writer Assistant')
    replicate_api = st.text_input('Enter Replicate API token:', type='password')
    if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
        st.warning('Please enter your Replicate API token.', icon='⚠️')
        st.markdown(
            "**Don't have an API token?** Head over to [Replicate]((link unavailable)) to sign up for one.")

    os.environ['REPLICATE_API_TOKEN'] = replicate_api
    st.subheader("Adjust model parameters")
    temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=5.0, value=0.7, step=0.01)
    top_p = st.sidebar.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant",
                                  "content": "Hi. I'm your Content Writer Assistant. How can I help you with your writing today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    if message["role"] in icons:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            st.write(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant",
                                  "content": "Hi. I'm your Content Writer Assistant. How can I help you with your writing today?"}]

st.sidebar.button('Clear chat history', on_click=clear_chat_history)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to ensure we're not sending too much text to the model."""
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b", rust_remote_code=True, trust_remote_code=True)

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt."""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

# Function for generating Content Writer Assistant response
def generate_content_response():
    prompt = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("user\n" + dict_message["content"] + "")
        else:
            prompt.append("assistant\n" + dict_message["content"] + "")

    prompt.append("assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)

    if get_num_tokens(prompt_str) >= 3072:
        st.error("Conversation length too long. Please keep it under 3072 tokens.")
        st.button('Clear chat history', on_click=clear_chat_history, key="clear_chat_history")
        st.stop()

    for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                                  input={"prompt": prompt_str,
                                         "prompt_template": r"{prompt}",
                                         "temperature": temperature,
                                         "top_p": top_p,
                                         }):
        yield str(event)

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="✏️"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="./Snowflake_Logomark_blue.svg"):
        response = generate_content_response()
        full_response = st.write_stream(response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
