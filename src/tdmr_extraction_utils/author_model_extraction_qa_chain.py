import os
from pathlib import Path

from src.utils import create_dir_if_not_exists, save_dict_to_json

MODEL_NAME = "gpt-4o"

# Load documents
import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.utils import read_json

load_dotenv()

# Define paths
FAISS_DB_PATH = "faiss_db"


# Define your desired data structure.
class Response(BaseModel):
    extracted_models_approaches: list[str] = Field(
        description="A list of models/approaches retireved from documents"
    )


# Load documents (Markdown files)
def load_documents(folder_path: str):
    if folder_path.endswith(".json"):
        return load_documents_from_extracted_dict(
            folder_path, str(Path(folder_path).parent).split("/")[-1]
        )
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".md"):  # Only process .md files
            file_path = os.path.join(folder_path, file_name)
            loader = TextLoader(file_path, metadata={"file_name": file_name})
            documents.extend(loader.load())
    return documents


def load_documents_from_extracted_dict(
    extracted_sections_file_path: str, file_name: str
) -> list[Document]:
    extracted_sections = read_json(Path(extracted_sections_file_path))
    documents_with_metadata = []
    for section in extracted_sections:
        documents_metadata = {"file_name": file_name, "section": section}
        documents_with_metadata.append(
            Document(
                page_content=extracted_sections[section], metadata=documents_metadata
            )
        )
    return documents_with_metadata


# Check if a document is new based on metadata
def get_new_documents(documents, existing_filenames):
    new_documents = []
    for doc in documents:
        if doc.metadata["file_name"] not in existing_filenames:
            new_documents.append(doc)
    return new_documents


def load_vector_store(folder_path):
    documents = load_documents(folder_path)

    # Load or create the FAISS database
    if os.path.exists(FAISS_DB_PATH):
        print("Loading existing FAISS database...")
        vector_store = FAISS.load_local(
            FAISS_DB_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True
        )
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

    return vector_store


# Main function to create and run the QA chain
def main(folder_path: str, output_dir_path):
    file_name = str(Path(folder_path).parent).split("/")[-1]
    # Load documents
    vector_store = load_vector_store(folder_path)
    # Create the retrieval-based QA chain
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 6, "filter": {"file_name": file_name}}
    )
    llm = ChatOpenAI(model=MODEL_NAME)  # "gpt-3.5-turbo"
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
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    # Run the QA chain interactively
    query = "what's the name of approaches/models that the authors of this paper introduced? Please provide the names as they apper in results tables."

    result = qa_chain.invoke({"query": query, "filter": {"file_name": file_name}})
    print("\nAnswer:", result["result"])
    print("\nSources:")
    result_query = f"Having extracted authors approach/mode within this response:{result['result']} please extract metric and value of the metric which corresponds to this approach/model"
    result_response = qa_chain.invoke(
        {"query": result_query, "filter": {"file_name": file_name}}
    )
    print("\nAnswer:", result_response["result"])

    result_dict = {
        "authors_approach_response": result["result"],
        "authors_approach_result": result_response["result"],
    }
    save_dict_to_json(result_dict, Path(output_dir_path, file_name + ".json"))

    # for doc in result["source_documents"]:
    #     print("-", doc.metadata.get("file_name", "Unknown"))


if __name__ == "__main__":
    parsed_papers_without_table_content_dir = "parsing_experiments/15_12_2024_gpt-4o"
    parsed_papers_without_table_content = list(
        Path(parsed_papers_without_table_content_dir).iterdir()
    )
    author_model_extraction_dir_without_table_content = (
        f"author_model_extraction/from_each_section_{MODEL_NAME}"
    )
    create_dir_if_not_exists(Path(author_model_extraction_dir_without_table_content))

    # vector_store = load_vector_store()

    for paper_path in parsed_papers_without_table_content:
        paper_name_output_path = Path(
            f"{author_model_extraction_dir_without_table_content}/{paper_path.name}"
        )
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
        main(papers_section_text_path, paper_name_output_path)
