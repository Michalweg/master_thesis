import os
from pathlib import Path

from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.logger import logger
from src.openai_client import get_llm_model_response, get_openai_model_response
from src.parsers.llm_parser import (parse_model_response,
                                    send_request_to_the_model_with_ollama)
from src.parsers.docling_parsers import convert_pdf_into_md_using_docling
from src.parsers.marker_parser import parse_pdf_with_marker
from src.parsers.parser import extract_pdf_sections_content
from src.utils import (create_dir_if_not_exists, read_json,
                       read_markdown_file_content, save_dict_to_json,
                       save_str_as_markdown, save_str_as_txt_file)
from prompts.author_model_extraction import EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_SYSTEM_PROMPT, EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT
from typing import Dict, List, Union
from collections import Counter

MODEL_NAME = "openai-gpt-oss-120b"


class AuthorsModelResponse(BaseModel):
    extracted_model_approach_names: list[str] = Field(
        description="List of extracted model/approaches developed by authors of the paper. "
    )


def chunk_markdown_file(file_path: str, chunk_size: int) -> Union[Dict[str, str], None]:
    """
    Reads a markdown file and splits its content into a dictionary of numbered chunks.

    The content is split into chunks of approximately `chunk_size` words.

    Args:
        file_path (str): The path to the markdown (.md) file.
        chunk_size (int): The desired number of words per chunk.

    Returns:
        Union[Dict[str, str], None]: A dictionary where keys are consecutive integers
                                     and values are the text chunks. Returns None if
                                     the file is not found.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file at path '{file_path}' does not exist.")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split content into words and filter out empty strings
        words = content.split()

        chunks = {}
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks[str(len(chunks))] = chunk

        return chunks

    except IOError as e:
        print(f"Error reading the file: {e}")
        return None


def extract_author_model_prediction(
    markdown_file_path: str | Path,
    output_dir: str | Path,
    system_prompt: str,
    user_prompt: str,
    url: str = "http://localhost:11434/api/generate",
    model_name: str = "mistral",
):
    logger.info(f"Extracting author model using {model_name} ...")
    file_content = read_markdown_file_content(markdown_file_path)
    if file_content:
        prompt = PromptTemplate(template=user_prompt).format(section=file_content)

        try:
            model_response = get_llm_model_response(prompt, model_name=model_name, system_prompt=system_prompt, pydantic_object_structured_output=AuthorsModelResponse)
            print(model_response)
        except Exception as e:
            logger.error(str(e))
            model_response = send_request_to_the_model_with_ollama(
                prompt, model_name, url
            )

        if model_response:
            parsed_response_file_path = os.path.join(
                str(output_dir),
                Path(markdown_file_path).with_suffix(".json")
            )
            model_response = model_response if isinstance(model_response, dict) else model_response.model_dump()
            save_dict_to_json(model_response, parsed_response_file_path)
        else:
            logger.info(f"The model didn't generate any response")
            save_str_as_txt_file(
                txt_file_path=os.path.join(
                    str(output_dir),
                    Path(markdown_file_path).stem + f"len_{len(file_content)}" + ".txt",
                ),
                str_content=model_response,
            )
        logger.info(f"Extracting author/model/approach with {model_name} done")
    else:
        logger.warning(
            f"There is no content in this markdown file: {markdown_file_path}"
        )


def combine_all_sections_based_json_into_one_file(
    output_dir_path: str, result_file_name: str = "author_model_approaches.json"
) -> None:
    output_sections_dict_list: list[dict] = []
    counter = Counter()
    for section_file in Path(output_dir_path).iterdir():
        if section_file.suffix == ".json" and section_file.name != "author_model_approaches.json":
            section_result = read_json(section_file)
            if section_result["extracted_model_approach_names"]:
                output_sections_dict_list.append(
                    {
                        section_file.name.split("len")[0]: section_result[
                            "extracted_model_approach_names"
                        ]
                    }
                )
                counter.update(section_result["extracted_model_approach_names"])
    try:
        output_sections_dict_list.append({"most_common_model_name": counter.most_common(1)[0][0]})
    except IndexError:
        logger.warning("No most frequent in result")
    save_dict_to_json(
        output_sections_dict_list, os.path.join(output_dir_path, result_file_name)
    )


if __name__ == "__main__":
    user_marker: bool = False
    author_model_approach_experiment_dir_path = (
        "extending_results_extraction_with_author_approach"
    )
    papers_dir = os.path.join(author_model_approach_experiment_dir_path, "papers")
    papers_to_analyze: list = list(Path(papers_dir).iterdir())

    # Setting up and creating output dir
    author_model_extraction_dir_without_table_content = f"author_model_extraction/{MODEL_NAME}/from_each_section_22_09"
    create_dir_if_not_exists(Path(author_model_extraction_dir_without_table_content))
    already_processed_file = [f.name for f in Path(author_model_extraction_dir_without_table_content).iterdir()]

    for paper_path in tqdm(papers_to_analyze):

        if paper_path.stem in already_processed_file:
            logger.info(f"File: {paper_path} already processed ...")
            continue

        if paper_path.suffix != ".pdf":
            logger.warning(f"Bad files suffix: {paper_path.name}")
            continue

        logger.info(f"Processing: {paper_path}")

        # Setting up and creating an output di for specific paper
        paper_name_output_path = Path(
            f"{author_model_extraction_dir_without_table_content}/{paper_path.stem}"
        )
        create_dir_if_not_exists(paper_name_output_path)

        # Reading extracted_text_dict.json if exists, otherwise create it on the spot
        papers_section_text_path = os.path.join(
            paper_path,
            "extracted_text_dict.json",
        )

        if not Path(papers_section_text_path).exists():
            if user_marker:
            # Create this file by parsing the file with Marker and then extract section out of it.
                marker_output_dir = f"{author_model_approach_experiment_dir_path}/{paper_path.stem}/marker_output"
                markdown_file_path = parse_pdf_with_marker(
                    str(paper_path), marker_output_dir
                )
                papers_section_text = extract_pdf_sections_content(markdown_file_path)
            else:
                md_file_path = convert_pdf_into_md_using_docling(paper_path)
                papers_section_text = chunk_markdown_file(md_file_path, 4000)
        else:
            papers_section_text = read_json(Path(papers_section_text_path))

        # Iterating through each section
        for section in papers_section_text:
            if papers_section_text[section]:
                save_str_as_markdown(f"{section}.md", papers_section_text[section])
                extract_author_model_prediction(
                    markdown_file_path=f"{section}.md",
                    output_dir=paper_name_output_path,
                    model_name=MODEL_NAME,
                    system_prompt=EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_SYSTEM_PROMPT,
                    user_prompt=EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT
                )
                os.remove(f"{section}.md")
            else:
                logger.warning(
                    f"For this section: {section} no content could be extracted thus no model/approach"
                )


        # Saving all extracted results for given paper
        combine_all_sections_based_json_into_one_file(output_dir_path=str(paper_name_output_path))
