import os
from pathlib import Path

import pandas as pd
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
from src.utils import (create_dir_if_not_exists, extract_tables_from_markdown,
                       read_json,
                       read_markdown_file_content, save_dict_to_json,
                       save_str_as_markdown, save_str_as_txt_file)
from prompts.author_model_extraction import (
    EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_SYSTEM_PROMPT,
    EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT,
    EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT_WITH_TABLE_CONTEXT,
    EXTRACT_MODEL_NAMES_FROM_TABLE_SYSTEM_PROMPT,
    EXTRACT_MODEL_NAMES_FROM_TABLE_USER_PROMPT,
)
from typing import Dict, List, Union
from collections import Counter
from src.tdmr_extraction_utils.utils import chunk_markdown_file

MODEL_NAME = "openai-gpt-oss-120b"


class AuthorsModelResponse(BaseModel):
    extracted_model_approach_names: list[str] = Field(
        description="List of extracted model/approaches developed by authors of the paper. "
    )


class TableModelNamesResponse(BaseModel):
    model_approach_names: list[str] = Field(
        description="List of all model/approach names found in the table."
    )



def extract_model_names_from_table(
    table_df: pd.DataFrame,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
) -> list[str]:
    try:
        table_md = table_df.to_markdown()
        prompt = PromptTemplate(template=user_prompt).format(table=table_md)
        response = get_llm_model_response(
            prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            pydantic_object_structured_output=TableModelNamesResponse,
        )
        if isinstance(response, TableModelNamesResponse):
            return response.model_approach_names
        elif isinstance(response, dict) and "model_approach_names" in response:
            return response["model_approach_names"]
        return []
    except Exception as e:
        logger.error(f"Error extracting model names from table: {e}")
        return []


def extract_all_model_names_from_tables(
    md_file_path: str | Path,
    model_name: str,
) -> list[str]:
    logger.info(f"Extracting model names from tables in {md_file_path}...")
    tables = extract_tables_from_markdown(str(md_file_path))
    all_model_names: list[str] = []
    for table_df in tables:
        names = extract_model_names_from_table(
            table_df,
            system_prompt=EXTRACT_MODEL_NAMES_FROM_TABLE_SYSTEM_PROMPT,
            user_prompt=EXTRACT_MODEL_NAMES_FROM_TABLE_USER_PROMPT,
            model_name=model_name,
        )
        all_model_names.extend(names)
    deduplicated = sorted(set(all_model_names))
    logger.info(f"Found {len(deduplicated)} unique model names from tables: {deduplicated}")
    return deduplicated


def extract_author_model_prediction(
    markdown_file_path: str | Path,
    output_dir: str | Path,
    system_prompt: str,
    user_prompt: str,
    url: str = "http://localhost:11434/api/generate",
    model_name: str = "mistral",
    table_model_names: list[str] | None = None,
):
    logger.info(f"Extracting author model using {model_name} ...")
    file_content = read_markdown_file_content(markdown_file_path)
    if file_content:
        if table_model_names:
            prompt = PromptTemplate(template=user_prompt).format(
                section=file_content,
                table_model_names=", ".join(table_model_names),
            )
        else:
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
    user_marker: bool = False # Otherwise use docling for that purpose
    author_model_approach_experiment_dir_path = (
        "extending_results_extraction_with_author_approach"
    )
    papers_dir = "custom_dataset_papers/dbpedia/LC-QuAD v1" # os.path.join(author_model_approach_experiment_dir_path, "papers")
    papers_to_analyze: list = list(Path(papers_dir).iterdir())

    # Setting up and creating output dir
    author_model_extraction_dir_without_table_content = f"author_model_extraction/{MODEL_NAME}/08_11_custom_dataset_dbpedia"
    create_dir_if_not_exists(Path(author_model_extraction_dir_without_table_content))
    already_processed_file = [f.name for f in Path(author_model_extraction_dir_without_table_content).iterdir() if f.suffix == ".pdf"]

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

        full_md_file_path = None
        if not Path(papers_section_text_path).exists():
            if user_marker:
            # Create this file by parsing the file with Marker and then extract section out of it.
                marker_output_dir = f"{author_model_approach_experiment_dir_path}/{paper_path.stem}/marker_output"
                markdown_file_path = parse_pdf_with_marker(
                    str(paper_path), marker_output_dir
                )
                full_md_file_path = markdown_file_path
                papers_section_text = extract_pdf_sections_content(markdown_file_path)
            else:
                md_file_path = convert_pdf_into_md_using_docling(paper_path)
                full_md_file_path = md_file_path
                papers_section_text = chunk_markdown_file(md_file_path, 4000)
        else:
            papers_section_text = read_json(Path(papers_section_text_path))

        # Extract model names from tables if we have the markdown file
        table_model_names = []
        if full_md_file_path:
            table_model_names = extract_all_model_names_from_tables(full_md_file_path, MODEL_NAME)

        # Choose the appropriate user prompt based on whether table names were found
        if table_model_names:
            active_user_prompt = EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT_WITH_TABLE_CONTEXT
        else:
            active_user_prompt = EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT

        # Iterating through each section
        for section in papers_section_text:
            if papers_section_text[section]:
                save_str_as_markdown(f"{section}.md", papers_section_text[section])
                extract_author_model_prediction(
                    markdown_file_path=f"{section}.md",
                    output_dir=paper_name_output_path,
                    model_name=MODEL_NAME,
                    system_prompt=EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_SYSTEM_PROMPT,
                    user_prompt=active_user_prompt,
                    table_model_names=table_model_names if table_model_names else None,
                )
                os.remove(f"{section}.md")
            else:
                logger.warning(
                    f"For this section: {section} no content could be extracted thus no model/approach"
                )


        # Saving all extracted results for given paper
        combine_all_sections_based_json_into_one_file(output_dir_path=str(paper_name_output_path))
