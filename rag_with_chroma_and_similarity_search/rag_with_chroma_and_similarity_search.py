# This example demonstrates how to use LangChain with Chroma and OpenAI's GPT-4o-mini model for a retrieval-augmented generation (RAG) task.
# The code loads a CSV file, splits the text into chunks, creates embeddings, and stores them in a Chroma vector store.
# It then retrieves relevant documents based on a query and generates a response using the LLM.
import os
import httpx
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(verify=False)
)

loader = CSVLoader('rag.csv')
data = loader.load()

# Split the text into chunks using RecursiveCharacterTextSplitter
# This is a text splitter that splits the text into smaller chunks based on the specified separators and chunk size.
splitter = RecursiveCharacterTextSplitter(
  separators=["\n\n", "\n", " ", ""],
  chunk_size=300,
  chunk_overlap=50
)

docs = splitter.split_documents(data) 

embedding_function = OpenAIEmbeddings(
  api_key=OPENAI_API_KEY,
  model='text-embedding-3-small',
  http_client=httpx.Client(verify=False)
)

# Create a Chroma vector store from the documents and embeddings
# The Chroma vector store is a database that stores the document embeddings for efficient retrieval.
# The `persist_directory` parameter specifies the directory where the embeddings will be stored.
vectorstore = Chroma.from_documents(
  docs,
  embedding=embedding_function,
  persist_directory="./docs"
)

# Persist the vector store to disk
retriever = vectorstore.as_retriever(
  search_type="similarity",
  search_kwargs={"k": 2}
)

message = """
Review and fix the following TechStack marketing copy with the following guidelines in consideration:

Guidelines:
{guidelines}

Copy:
{copy}

Fixed Copy:
"""

prompt_template = ChatPromptTemplate.from_messages([("human", message)])

# Create a retrieval-augmented generation (RAG) chain using the retriever and LLM
# The RAG chain retrieves relevant documents based on the query and generates a response using the LLM.
rag_chain = ({"guidelines": retriever, "copy": RunnablePassthrough()}
            | prompt_template
            | llm)

response = rag_chain.invoke("Here at TechStack, our users are the best in the world!")
print(response.content)