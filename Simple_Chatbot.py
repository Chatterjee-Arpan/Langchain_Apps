## Conversational Q&A Chatbot
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage,AIMessage
from langchain_openai import ChatOpenAI

#Streamlit UI
st.set_page_config(page_title="Conversational Q&A Chatbot", page_icon="🤖")
st.header("What's on your mind today?")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment or .env")
else:
    os.environ["OPENAI_API_KEY"] = api_key

chat=ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are a helpful fact-checking assistant.")
    ]

##Function to get response from the chatbot
def get_chatbot_response(question):

    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer=chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))

    return answer.content

input=st.text_input("Ask me anything:", placeholder="Type your question here...", key="input")
response = get_chatbot_response(input)

submit=st.button("Ask it!")

## if button is clicked
if submit:
    st.subheader("My thoughts on this:")
    st.write(response)