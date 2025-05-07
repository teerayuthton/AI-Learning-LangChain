# RAG with Chroma and Similarity search
- This example demonstrates how to use LangChain with Chroma and OpenAI's GPT-4o-mini model for a retrieval-augmented generation (RAG) task.
- Required install
```
pip install langchain_chroma
```

## Step
1. Initialize the OpenAI LLM with the specified model and API key `llm = ChatOpenAI`
2. Load the CSV file using the CSVLoader `loader = CSVLoader`
3. Split the text into chunks using RecursiveCharacterTextSplitter `splitter = RecursiveCharacterTextSplitter`
4. Create an embedding function using OpenAIEmbeddings `embedding_function = OpenAIEmbeddings`
5. Create a Chroma vector store from the documents and embeddings `vectorstore = Chroma.from_documents`
6. Persist the vector store to disk `retriever = vectorstore.as_retriever`
7. Create a prompt template for the LLM `prompt_template = ChatPromptTemplate.from_messages`
8. Create a retrieval-augmented generation (RAG) chain using the retriever and LLM `rag_chain =(...)`

<img width="521" alt="Screenshot 2568-05-05 at 22 13 35" src="https://github.com/user-attachments/assets/f171756f-ad7b-4cdd-807d-0c3c791ce503" />


The result is
```
Here at TechStack, our users, whom we proudly refer to as techies, are the best in the world!
```
