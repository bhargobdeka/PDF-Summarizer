# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# set openai_api key
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Create OpenAI LLM object
llm = OpenAI(temperature=0.9,verbose=True)

# Create OpenAI Embeddings object
embeddings = OpenAIEmbeddings()

# Create and load PDF Loader
loader = PyPDFLoader('../docs/agvi.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='paperreport')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="paper_report",
    description="AGVI Paper Report",
    vectorstore=store
)

# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

st.title('AGVI paper Q/A')

# Create a text input box for the user
prompt = st.text_input('Ask a question about the AGVI paper')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 




