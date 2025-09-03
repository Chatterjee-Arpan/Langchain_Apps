## Integrate our code OpenAI API with LangChain
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain
# from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory # This import is for memory buffer

## Loading API Key from .env file

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment or .env")
else:
    os.environ["OPENAI_API_KEY"] = api_key


## Initialize Streamlit framework
st.title("Celebrity Search Results")
input_text = st.text_input("Enter the Celebrity that you want to know about:")

## Prompt Template designing
first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about celebrity {name}."
)

### Memory Buffer Initialization
person_memory = ConversationBufferMemory(input_key="name", memory_key="chat_history")
dob_memory = ConversationBufferMemory(input_key="title", memory_key="dob_history")
events_memory = ConversationBufferMemory(input_key="dob", memory_key="events_history")

## OpenAI LLM initialization
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
llm_chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose =True, output_key="title", memory=person_memory)

## 2nd Prompt Template designing
second_input_prompt = PromptTemplate(
    input_variables=["title"],
    template="When was {title} born?"
)
llm_chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose =True, output_key="dob", memory=dob_memory)

third_input_prompt = PromptTemplate(
    input_variables=["dob"],
    template="Mention 5 major events that happened around {dob} in the world."
)

llm_chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose =True, output_key="events", memory=events_memory)

## Creating a Sequential Chain
# llm_chains = SimpleSequentialChain(chains=[llm_chain, llm_chain2], verbose=True)
## SimpleSequentialChain only provides the final output, while SequentialChain allows access to all outputs

llm_chains = SequentialChain(chains=[llm_chain, llm_chain2, llm_chain3], input_variables=["name"], output_variables=["title", "dob", "events"], verbose=True)

if input_text:
    #resp = llm_chains.run(input_text) #for SimpleSequentialChain
    resp = llm_chains({"name": input_text})
    st.write(resp)

    with st.expander("Person Name"):
        st.info(person_memory.buffer)

    with st.expander("Major Events"):
        st.info(events_memory.buffer)