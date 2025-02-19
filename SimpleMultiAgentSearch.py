import streamlit as st
import matplotlib.pyplot as plt
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

from langchain_openai import ChatOpenAI

from typing import List, TypedDict, Annotated, Literal, Optional, Union
from langgraph.graph import StateGraph, END


import sqlite3
import operator
import os

#YOUR API KEYS
os.environ["OPENAI_API_KEY"] = "INSERT YOUR API KEY FOR OPENAI"
os.environ["TAVILY_API_KEY"] = "INSERT YOUR API KEY FOR TAVILY"

#LLM Initialisation
model = ChatOpenAI(model="gpt-4o-mini")





class SearchAgentState(TypedDict):
    query: str
    answer: str
    context: Annotated[list, operator.add]

def begin_search(state):
    st.write("Searching...")
    return state

def search_web(state : SearchAgentState) -> SearchAgentState:
    """ Retrieve search results from the web """
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['query'])

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

def search_wiki(state : SearchAgentState) -> SearchAgentState:
    """ Retrieve search results from the Wikipedia """
    search_docs = WikipediaLoader(query=state['query'], 
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

def generate_answer(state : SearchAgentState) -> SearchAgentState:
    """ Generate answer from the context """

    context = state["context"]
    question = state["query"]

    system_prompt = f"Answer the following question using the provided context: {question}\n\n. Context for answering:\n\n{context}"
    answer = model.invoke([SystemMessage(content = system_prompt)] + [HumanMessage(content = "Answer the question!")])

    return {"answer": answer}

#graph builder
builder = StateGraph(SearchAgentState)
builder.add_node("start", begin_search)
builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wiki)
builder.add_node("generate_answer", generate_answer)

builder.set_entry_point("start")
builder.add_edge("start", "search_wikipedia")
builder.add_edge("start", "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()

#Streamlit UI
def main():
    st.title("Multi-agent search system based on LangGraph")
    st.write("Hello, username! Please input your search query below:")

    query = st.text_input("Your query", "")

    if st.button("Start search"):
        if not query.strip():
            st.warning("Please, provide your query!")
        else:
            with st.spinner("Searching..."):
                initial_state = {"query": query, "context": []}
                result_state = graph.invoke(initial_state)
                answer = result_state.get("answer")
                if answer:
                    st.subheader("Answer:")
                    st.write(answer.content)
                else:
                    st.error("Answer has not received! Try new query!")

if __name__ == "__main__":
    main()
