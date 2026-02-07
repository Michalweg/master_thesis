import os
from io import StringIO
from pathlib import Path

import pandas as pd
from langchain.prompts import PromptTemplate

from src.logger import logger
from src.openai_client import get_llm_model_response, get_openai_model_response
from src.tdmr_extraction_utils.data_models import (
    PREPARED_DICT_INTO_STR_TEMPLATE, PreparedTable, TableDecisionResponse)
from src.utils import read_json, save_dict_to_json
from tqdm import tqdm
from typing import Dict, Union

def prepare_extracted_tables_for_experiment(
    extracted_tables_dict_object: list[dict],
) -> list[PreparedTable]:
    prepared_dicts = []
    for i, table_object in enumerate(extracted_tables_dict_object):
        csv_data = StringIO(table_object["data"])
        table = pd.read_csv(csv_data)
        table_caption = table_object["caption"]
        table_id = i + 1
        prepared_dict = PreparedTable(
            table=table.to_dict("records"),
            table_caption=table_caption,
            table_id=table_id,
        )
        prepared_dicts.append(prepared_dict)
    return prepared_dicts


def merge_results_by_keys(dict_list):
    merged = {}

    for d in dict_list:
        key = (
            d.get("Task").strip("'\""),
            d.get("Dataset").strip("'\""),
            d.get("Metric").strip("'\""),
        )
        result = d.get("Result")

        if key not in merged:
            # Initialize a base dictionary with an empty result list
            merged[key] = {
                "Task": d.get("Task").strip("'\""),
                "Dataset": d.get("Dataset").strip("'\""),
                "Metric": d.get("Metric").strip("'\""),
                "Result": [],
            }
            # Preserve Model field if it exists
            if d.get("Model"):
                merged[key]["Model"] = d.get("Model").strip("'\"")

        # Add result if it exists and is not None
        if result is not None:
            # If it's a list, extend; otherwise, append as a single item
            if isinstance(result, list):
                merged[key]["Result"].extend(result)
            else:
                merged[key]["Result"].append(result)

    # Return as a list of merged dictionaries
    return list(merged.values())


def formate_prepared_dicts_into_str(prepared_dicts: list[PreparedTable]) -> str:
    dicts_into_str = ""

    for prepared_dict in prepared_dicts:
        dicts_into_str += PREPARED_DICT_INTO_STR_TEMPLATE.format(
            table_markdown=prepared_dict.dataframe.head(20).to_markdown(),
            table_caption=prepared_dict.table_caption,
            table_id=prepared_dict.table_id,
        )
        dicts_into_str += "\n"
    return dicts_into_str


def prepare_prompt_for_table_choosing(
    extracted_triplet: dict,
    prepared_dicts_str: str,
    prompt_template: str,
    system_prompt: str = "",
) -> tuple[str, str]:
    if system_prompt:
        prompt = PromptTemplate(
            input_variables=["table_markdown", "table_caption", "table_id"],
            template=prompt_template,
        ).format(triplet=extracted_triplet, tables_data=prepared_dicts_str)
    else:
        prompt_template_combined = system_prompt + prompt_template
        prompt = PromptTemplate(
            input_variables=["table_markdown", "table_caption", "table_id"],
            template=prompt_template_combined,
        ).format(triplet=extracted_triplet, tables_data=prepared_dicts_str)
        system_prompt = ""
    return prompt, system_prompt


def pick_optimal_source_table_for_given_triplet(
    extracted_triplet: dict,
    prepared_dicts: list[PreparedTable],
    prompt_template: str,
    result_object: TableDecisionResponse,
    model_name: str,
    structured_output: bool = True,
    system_prompt: str = "",
) -> dict:
    prepared_dicts_into_str = formate_prepared_dicts_into_str(prepared_dicts)
    prompt, system_prompt = prepare_prompt_for_table_choosing(
        extracted_triplet,
        prepared_dicts_into_str,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
    )
    if structured_output:
        response = get_llm_model_response(
            prompt=prompt,
            pydantic_object_structured_output=result_object,
            system_prompt=system_prompt,
            model_name=model_name,
        )
        if response:
            if not isinstance(response, dict):
                response = response.model_dump()
    else:
        response = get_llm_model_response(
            prompt=prompt,
            pydantic_object_structured_output=None,
            system_prompt=system_prompt,
            model_name=model_name,
        )
    logger.info(f"Response for picking table: {response}")
    return response


def create_one_result_file(output_dir: Path):
    processed_dicts = []
    all_unique_papersInitial = set()
    for index, paper_dir in enumerate(list(output_dir.iterdir())):
        if paper_dir.is_dir():
            paper_result_file_path = os.path.join(
                paper_dir, paper_dir.name + "_tdmr_extraction.json"
            )
            paper_result_json = read_json(Path(str(paper_result_file_path)))
            for result_dict in paper_result_json:
                if isinstance(result_dict, dict):
                    result_dict.update({"PaperName": paper_dir.name})
                    all_unique_papersInitial.add(paper_dir.name)
                    result_keyword = "Result" if "Result" in result_dict else "Results"
                    if result_keyword in result_dict:
                        stay_dict = {
                            k: v for k, v in result_dict.items() if k != result_keyword
                        }
                        if isinstance(result_dict[result_keyword], dict):
                            new_dicts = []
                            for k, v in result_dict[result_keyword].items():
                                new_result_dict = stay_dict.copy()
                                new_result_dict.update({"Result": v})
                                new_dicts.append(new_result_dict)
                            processed_dicts.extend(new_dicts)

                        else:
                            processed_dicts.append(result_dict)
                    else:
                        processed_dicts.append(result_dict)
                else:
                    print(result_dict)
    save_dict_to_json(
        processed_dicts,
        os.path.join(output_dir, "processed_tdmr_extraction_test_papers.json"),
    )


def _is_valid_paper_directory(paper_dir: Path) -> bool:
    """Check if the directory is a valid paper directory (not DS_Store or other system files)."""
    return paper_dir.is_dir() and "ds_store" not in paper_dir.name.lower()


def _normalize_result_key(result_dict: dict) -> tuple[dict, bool]:
    """
    Normalize 'Results' key to 'Result' if needed.
    Returns the modified dict and whether normalization occurred.
    """
    if "Result" in result_dict:
        return result_dict, False
    if "Results" in result_dict:
        result_dict["Result"] = result_dict.pop("Results")
        return result_dict, True
    return result_dict, False


def _process_list_result(result_dict: dict, paper_name: str) -> dict | None:
    """
    Process a result dict where the Result value is a list.
    Returns the processed dict or None if processing fails.
    """
    results_list = result_dict["Result"]

    if not results_list:
        return None

    if len(results_list) == 1:
        results_list = str(results_list[0]).split(", ")
        result_dict["Result"] = results_list

    try:
        return {
            **result_dict,
            "Result": str(max(results_list)),
            "ResultList": [str(x) for x in results_list],
        }
    except TypeError:
        return _handle_mixed_type_results(result_dict, results_list, paper_name)


def _handle_mixed_type_results(result_dict: dict, results_list: list, paper_name: str) -> dict | None:
    """Handle results list with mixed types (some numeric, some not)."""
    valid_results = [r for r in results_list if isinstance(r, (int, float))]
    invalid_results = [r for r in results_list if not isinstance(r, (int, float))]

    logger.warning(
        f"Invalid elements from results list extracted: {invalid_results} from the file: {paper_name}"
    )

    if not valid_results:
        logger.error(
            f"No valid results found for this list: {results_list} from this file: {paper_name}"
        )
        return None

    return {
        **result_dict,
        "Result": str(max(valid_results)),
        "ResultList": [str(x) for x in results_list],
    }


def _process_single_result_dict(
    result_dict: dict,
    paper_name: str,
    stats: dict
) -> tuple[dict | None, dict | None]:
    """
    Process a single result dictionary.
    Returns (paper_result_dict, processed_dict) tuple, either can be None.
    """
    if not isinstance(result_dict, dict):
        logger.warning(
            f"This result object is totally off: {result_dict} in the {paper_name} directory"
        )
        print(result_dict)
        return None, None

    result_dict["PaperName"] = paper_name
    result_dict, was_normalized = _normalize_result_key(result_dict)

    if was_normalized:
        stats["results_instead_of_result"] += 1

    if "Result" not in result_dict:
        stats["without_result"] += 1
        return None, None

    result_value = result_dict["Result"]

    if isinstance(result_value, dict):
        stats["dicts_instead_of_lists"] += 1
        logger.error(
            f"From the file: {paper_name} we got dict instead of a list: {result_value}"
        )
        return None, None

    if isinstance(result_value, list):
        paper_result_dict = _process_list_result(result_dict, paper_name)
        if paper_result_dict:
            return paper_result_dict, result_dict
        return None, result_dict

    if isinstance(result_value, (float, str)):
        result_dict["Result"] = str(result_value)
        return result_dict, result_dict

    logger.error(f"This is not handled yet!: {result_value}")
    return None, None


def create_one_result_file_for_evaluation_purpose(output_dir: Path):
    """
    Create a consolidated result file for evaluation purposes.

    Processes all paper directories in output_dir and creates a JSON file with structure:
    {paper_name.pdf: {"normalized_output": [list of extracted triplets]}}
    """
    processed_dicts = []
    evaluation_result_dict = {}
    stats = {
        "without_result": 0,
        "results_instead_of_result": 0,
        "dicts_instead_of_lists": 0,
    }

    paper_dirs = [d for d in output_dir.iterdir() if _is_valid_paper_directory(d)]

    for paper_dir in tqdm(paper_dirs):
        paper_results = []
        result_file_path = paper_dir / f"{paper_dir.name}_tdmr_extraction.json"
        paper_result_json = read_json(result_file_path)
        paper_result_json = merge_results_by_keys(paper_result_json)

        for result_dict in paper_result_json:
            paper_result, processed_result = _process_single_result_dict(
                result_dict, paper_dir.name, stats
            )

            if paper_result:
                paper_results.append(paper_result)
            if processed_result:
                processed_dicts.append(processed_result)

        logger.info(f"Analysis of a dir called: {paper_dir.name} is done!")
        evaluation_result_dict[f"{paper_dir.name}.pdf"] = {
            "normalized_output": paper_results
        }

    logger.info(
        f"No. of entries with dicts not lists: {stats['dicts_instead_of_lists']} \n"
        f"No. of tuples without result data: {stats['without_result']} \n"
        f"No. of tuples with results and not result: {stats['results_instead_of_result']}"
    )

    output_path = output_dir / "processed_tdmr_extraction_test_papers_evaluation_valid_results.json"
    save_dict_to_json(evaluation_result_dict, str(output_path))


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