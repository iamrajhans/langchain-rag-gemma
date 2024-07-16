from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough



# use nomic embedding
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

# use Chroma vector store
db = Chroma(persist_directory="./db-fin-transcript",
            embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": 3}
)

# use gemma2
local_llm = 'gemma2'

llm = ChatOllama(model=local_llm,
                 keep_alive="5h",
                 max_tokens=512,
                 temperature=0)

# Create prompt template
template = """<bos><start_of_turn>user\nAnswer the question based only on the following context and extract out a meaningful answer. \
Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
If the context doen't contain the answer, just respond that you are unable to find an answer. \

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model\n
ANSWER:"""

prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain using LCEL with pro
# pt printing and streaming output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Function to ask questions
def ask_question(question):
    print("Answer:\n\n", end=" ", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk.content, end="", flush=True)
    print("\n")

# Example usage
if __name__ == "__main__":
    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        answer = ask_question(user_question)
