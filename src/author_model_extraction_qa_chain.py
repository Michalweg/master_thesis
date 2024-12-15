import os
from pathlib import Path

from src.utils import create_dir_if_not_exists

MODEL_NAME = "gpt-4o"

# Load documents
import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from src.utils import read_json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate


load_dotenv()

# Define paths
FAISS_DB_PATH = "faiss_db"

# Define your desired data structure.
class Response(BaseModel):
    extracted_models_approaches: list[str] = Field(description="A list of models/approaches retireved from documents")

# Load documents (Markdown files)
def load_documents(folder_path: str):
    if folder_path.endswith('.json'):
        return load_documents_from_extracted_dict(folder_path, str(Path(folder_path).parent).split("/")[-1])
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".md"):  # Only process .md files
            file_path = os.path.join(folder_path, file_name)
            loader = TextLoader(file_path, metadata={"file_name": file_name})
            documents.extend(loader.load())
    return documents

def load_documents_from_extracted_dict(extracted_sections_file_path: str, file_name: str) -> list[Document]:
    extracted_sections = read_json(Path(extracted_sections_file_path))
    documents_with_metadata = []
    for section in extracted_sections:
        documents_metadata = {"file_name": file_name, "section": section}
        documents_with_metadata.append(Document(page_content=extracted_sections[section], metadata=documents_metadata))
    return documents_with_metadata

# Check if a document is new based on metadata
def get_new_documents(documents, existing_filenames):
    new_documents = []
    for doc in documents:
        if doc.metadata["file_name"] not in existing_filenames:
            new_documents.append(doc)
    return new_documents

# Main function to create and run the QA chain
def main(folder_path: str):

    # Load documents
    documents = load_documents(folder_path)

    # Load or create the FAISS database
    if os.path.exists(FAISS_DB_PATH):
        print("Loading existing FAISS database...")
        vector_store = FAISS.load_local(FAISS_DB_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        existing_filenames = {
            doc.metadata["file_name"]
            for doc in vector_store.docstore._dict.values()
            if "file_name" in doc.metadata
        }
    else:
        print("Creating new FAISS database...")
        vector_store = None
        existing_filenames = set()

    # Identify new documents
    new_documents = get_new_documents(documents, existing_filenames)

    if new_documents:
        print(f"Adding {len(new_documents)} new documents to the database...")
        # Not splitting as it's not needed, each document represents a section from the paper
        # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        # new_texts = text_splitter.split_documents(new_documents)

        if vector_store is None:
            vector_store = FAISS.from_documents(new_documents, OpenAIEmbeddings())
        else:
            vector_store.add_documents(new_documents)

        # Save updated database
        vector_store.save_local(FAISS_DB_PATH)
    else:
        print("No new documents to add.")

    # Create the retrieval-based QA chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model=MODEL_NAME) # "gpt-3.5-turbo"
    output_parser = JsonOutputParser(pydantic_object=Response)
    PROMPT_TEMPLATE = PromptTemplate(
        input_variables=["query"],
        template=(
            "You are an expert in extracting information from text. Given the following context, "
            "please identify and list all the names of approaches/models that the authors of this paper introduced? Please provide the names as they apper in results tables."
            "Provide your response in the required JSON format.\n\n"
            "Context:\n{context}\n\n"
            "Query:\n{query}\n\n"
            "{format_instructions}"
        ),
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    # Run the QA chain interactively
    query = "what's the name of approaches/models that the authors of this paper introduced? Please provide the names as they apper in results tables."
    'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"extracted_models_approaches": {"description": "A list of models/approaches retireved from documents", "items": {"type": "string"}, "title": "Extracted Models Approaches", "type": "array"}}, "required": ["extracted_models_approaches"]}\n```'

    result = qa_chain.invoke(query)
    print("\nAnswer:", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("file_name", "Unknown"))


if __name__ == "__main__":
    parsed_papers_without_table_content_dir = "parsing_experiments/15_12_2024_gpt-4o"
    parsed_papers_without_table_content = list(Path(parsed_papers_without_table_content_dir).iterdir())
    author_model_extraction_dir_without_table_content = f"author_model_extraction/from_each_section_{MODEL_NAME}"
    create_dir_if_not_exists(Path(author_model_extraction_dir_without_table_content))

    for paper_path in parsed_papers_without_table_content:
        paper_name_output_path = Path(f"{author_model_extraction_dir_without_table_content}/{paper_path.name}")
        create_dir_if_not_exists(paper_name_output_path)
        papers_section_text_path = os.path.join(
            paper_path,
            "extracted_text_dict.json",
        )
        # marker_output_dir = os.path.join(
        #     paper_path,
        #     "marker_output",
        #     paper_path.name,
        # )
        main(papers_section_text_path)