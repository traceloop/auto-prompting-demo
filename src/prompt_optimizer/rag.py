import os
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.prompts import RichPromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.github import GithubRepositoryReader, GithubClient

from dotenv import load_dotenv

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

from pydantic import BaseModel

load_dotenv()

QA_PROMPT_KEY = "response_synthesizer:text_qa_template"


Traceloop.init()

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

github_client = GithubClient(github_token=os.environ["GITHUB_TOKEN"])

client = OpenAI()
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
)
chroma_client = chromadb.PersistentClient(path="db/chroma_data")
traceloop_docs = chroma_client.get_or_create_collection(
    "traceloop-docs", embedding_function=embedding_function
)
vector_store = ChromaVectorStore(chroma_collection=traceloop_docs)


index = VectorStoreIndex.from_vector_store(
    vector_store=ChromaVectorStore(chroma_collection=traceloop_docs),
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
)


@task()
def rephrase_as_query(question: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"Rewrite the following question as a concise 2 word query for a vector database: {question}",
            }
        ],
    )
    return response.choices[0].message.content


@workflow()
def query_rag(prompt_template: str, question: str):
    results = traceloop_docs.query(
        query_texts=[rephrase_as_query(question)], n_results=5
    )

    concatenated_docs = "\n\n".join(results["documents"][0])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt_template.format(
                    context=concatenated_docs, question=question
                ),
            },
        ],
    )
    return response.choices[0].message.content


def load_data():
    reader = GithubRepositoryReader(
        github_client=github_client,
        owner="traceloop",
        repo="docs",
        filter_file_extensions=(
            [".mdx"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
    )
    docs = reader.load_data(branch="main")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    index.storage_context.persist()
