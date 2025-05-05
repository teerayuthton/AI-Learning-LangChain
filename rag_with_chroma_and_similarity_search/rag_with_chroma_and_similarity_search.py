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

vectorstore = Chroma.from_documents(
  docs,
  embedding=embedding_function,
  persist_directory="./docs"
)

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

rag_chain = ({"guidelines": retriever, "copy": RunnablePassthrough()}
            | prompt_template
            | llm)

response = rag_chain.invoke("Here at TechStack, our users are the best in the world!")
print(response.content)