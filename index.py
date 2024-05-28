import os
from langchain_community.document_loaders import TextLoader

from process import load_docs
root_dir = "./solana-trading-bot"
docs = []
file_extensions = []

load_docs(root_dir,file_extensions)

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splitted_text = text_splitter.split_documents(docs)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

my_activeloop_org_id = os.getenv("ORG_ID")
my_activeloop_dataset_name = "chat_with_gh"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(splitted_text)


from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
# Create a retriever from the DeepLake instance
retriever = db.as_retriever()

# Set the search parameters for the retriever
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 100
retriever.search_kwargs["k"] = 10

# Create a ChatOpenAI model instance
model = ChatOpenAI()

# Create a RetrievalQA instance from the model and retriever
qa = RetrievalQA.from_llm(model, retriever=retriever)

# Return the result of the query
qa.run("What is the repository's name?")