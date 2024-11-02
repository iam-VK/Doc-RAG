import os 
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import CHUNK_SIZE, CHUNK_OVERLAP, MODEL

# Maps extensions to doc loaders
ext2loader = {
    ".csv": (CSVLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

def load_doc(file_path,print_report=False):
    file_extension = os.path.splitext(file_path)[1]

    if file_extension in ext2loader:
            
        loader_type, loader_args = ext2loader[file_extension]
        loader = loader_type(file_path, **loader_args)
        load = loader.load()

        if print_report:
            print(f"Number of pages: {len(load)}")
            print(f"Length of a page: {len(load[1].page_content)}")
            print("Content of a page:", load[1].page_content)
        
        return load

    raise ValueError(f" '{file_extension}' file type not supported")


def splitter(documents,print_report=False):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = splitter.split_documents(documents)

    if print_report:
        print(f"Number of chunks: {len(chunks)}")
        print(f"Length of a chunk: {len(chunks[1].page_content)}")
        print("Content of a chunk:", chunks[1].page_content)

    return chunks


def word_embeddings(chunks):
    embeddings = OllamaEmbeddings(model=MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore 


def retrieve(query,vectorstore):
    retriever = vectorstore.as_retriever()
    return retriever.invoke(query)


def add_docs():
    document = load_doc("Networks_notes.pdf", print_report=True)
    chunks = splitter(document, print_report=True)
    vectorstore = word_embeddings(chunks)

    retrieve(query="Preliminary design involves considering all the network requirements",vectorstore=vectorstore)

add_docs()