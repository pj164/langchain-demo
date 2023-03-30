import os
from dotenv import dotenv_values
from langchain.llms import OpenAI

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.indexes import VectorstoreIndexCreator

config = dotenv_values(".env")

os.environ["OPENAI_API_KEY"] =  config['OPENAI_API_KEY']

llm = OpenAI(temperature=0.9)
loader = UnstructuredFileLoader(config['file_path'])
index = VectorstoreIndexCreator().from_loaders([loader])

# query = "what is the procedure say about data safeguards"
query = "what are managers and supervisors responsible for?"

print(index.query(query))