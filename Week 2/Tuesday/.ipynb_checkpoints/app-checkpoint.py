import chainlit as cl
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import chainlit as cl

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """
Use the following pieces of context to answer the user's question.
Please respond as if you were Ken from the movie Barbie. Ken is a well-meaning but naive character who loves to Beach. He talks like a typical Californian Beach Bro, but he doesn't use the word "Dude" so much.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
You can make inferences based on the context as long as it still faithfully represents the feedback.

Example of your response should be:

```
The answer is foo
```

Begin!
----------------
{context}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate(messages=messages)
chain_type_kwargs = {"prompt": prompt}

@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"RetrievalQA": "Consulting The Kens"}
    return rename_dict.get(orig_author, orig_author)

@cl.on_chat_start
async def init():
    msg = cl.Message(content=f"Building Index...")
    await msg.send()

    # build FAISS index from csv
    loader = CSVLoader(file_path="./data/barbie.csv", source_column="Review_Url")
    data = loader.load()
    documents = text_splitter.transform_documents(data)
    store = LocalFileStore("./cache/")
    core_embeddings_model = OpenAIEmbeddings()
    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model, store, namespace=core_embeddings_model.model
    )
    # make async docsearch
    docsearch = await cl.make_async(FAISS.from_documents)(documents, embedder)

    chain = RetrievalQA.from_chain_type(
        ChatOpenAI(model="gpt-4", temperature=0, streaming=True),
        chain_type="stuff",
        return_source_documents=True,
        retriever=docsearch.as_retriever(),
        chain_type_kwargs = {"prompt": prompt}
    )

    msg.content = f"Index built!"
    await msg.send()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=False, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb], )

    answer = res["result"]
    source_elements = []
    visited_sources = set()

    # Get the documents from the user session
    docs = res["source_documents"]
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    for source in all_sources:
        if source in visited_sources:
            continue
        visited_sources.add(source)
        # Create the text element referenced in the message
        source_elements.append(
            cl.Text(content="https://www.imdb.com" + source, name="Review URL")
        )

    if source_elements:
        answer += f"\nSources: {', '.join([e.content.decode('utf-8') for e in source_elements])}"
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()
