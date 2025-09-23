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


def create_one_result_file_for_evaluation_purpose(output_dir: Path):
    processed_dicts = []
    evaluation_result_dict = (
        {}
    )  # Dict which has a structure required for evaluation this approach to the author's
    # With the structure of a key representing a paper name (including. pdf file) and the value "normalized_outputs" which is
    # a list of extracted triplets for given file
    triplets_without_result = 0
    no_of_triplets_with_results_instead_of_results = 0
    no_of_dicts_instead_of_lists = 0

    for index, paper_dir in tqdm(enumerate(list(output_dir.iterdir()))):
        if (
            paper_dir.is_dir() and "DS_store".lower() not in paper_dir.name.lower()
        ):  # Excluding all non-papers dirs as well as DS_store
            papers_proceed_dicts = []
            paper_result_file_path = os.path.join(
                paper_dir, paper_dir.name + "_tdmr_extraction.json"
            )
            paper_result_json = read_json(Path(str(paper_result_file_path)))
            # Exclude all potential duplicates here (with respect to TDM, Result is being excluded)
            paper_result_json = merge_results_by_keys(paper_result_json)
            for result_dict in paper_result_json:

                if isinstance(result_dict, dict):
                    result_dict.update({"PaperName": paper_dir.name})
                    result_keyword = "Result" if "Result" in result_dict else "Results"
                    if result_keyword not in result_dict:
                        triplets_without_result += 1
                        continue
                    if result_keyword != "Result":
                        result_dict["Result"] = result_dict.pop("Results")
                        no_of_triplets_with_results_instead_of_results += 1
                    if result_keyword in result_dict:
                        if isinstance(result_dict[result_keyword], dict):
                            no_of_dicts_instead_of_lists += 1
                            logger.error(
                                f"From the file: {paper_dir.name} we got dict instead of a list: {result_dict[result_keyword]}"
                            )
                            continue

                        elif isinstance(
                            result_dict[result_keyword], list
                        ):  # The object associated with "Result" key is not a dict
                            if result_dict[result_keyword]:
                                if len(result_dict[result_keyword]) == 1:
                                    valid_list_of_strings_with_results = result_dict[
                                        result_keyword
                                    ][0].split(", ")
                                    result_dict[result_keyword] = (
                                        valid_list_of_strings_with_results
                                    )
                                try:
                                    paper_result_dict = {
                                        **result_dict,
                                        **{
                                            "Result": str(
                                                max(result_dict[result_keyword])
                                            )
                                        },
                                        **{
                                            "ResultList": [
                                                str(x)
                                                for x in result_dict[result_keyword]
                                            ]
                                        },
                                    }
                                except TypeError:
                                    valid_results_list = []
                                    invalid_results_list = []
                                    for result in result_dict[result_keyword]:
                                        if isinstance(result, (int, float)):
                                            valid_results_list.append(result)
                                        else:
                                            invalid_results_list.append(result)
                                    logger.warning(
                                        f"Invalid elements from results list extracted: {invalid_results_list} from the file: {paper_dir.name}"
                                    )
                                    if valid_results_list:
                                        paper_result_dict = {
                                            **result_dict,
                                            **{"Result": str(max(valid_results_list))},
                                            **{
                                                "ResultList": [
                                                    str(x)
                                                    for x in result_dict[result_keyword]
                                                ]
                                            },
                                        }
                                    else:
                                        logger.error(
                                            f"No valid results found for this list: {result_dict[result_keyword]} from this file: {paper_dir.name}"
                                        )
                                        continue
                                papers_proceed_dicts.append(paper_result_dict)
                            processed_dicts.append(result_dict)

                        elif isinstance(
                            result_dict[result_keyword], float
                        ) or isinstance(
                            result_dict[result_keyword], str
                        ):  # The object associated with "Result" key is not a dict and not a list!
                            result_dict[result_keyword] = str(
                                result_dict[result_keyword]
                            )
                            papers_proceed_dicts.append(result_dict)
                            processed_dicts.append(result_dict)
                        else:
                            logger.error(
                                f"This is not handled yet!: {result_dict[result_keyword]}"
                            )

                    else:
                        logger.warning(
                            f"Neither 'result' nor 'results' were found withing the analyzed object: {result_dict}"
                        )
                        processed_dicts.append(result_dict)
                        papers_proceed_dicts.append(result_dict)
                else:
                    logger.warning(
                        f"This result object is totally off: {result_dict} in the {paper_dir.name} directory"
                    )
                    print(result_dict)

            logger.info(f"Analysis of a dir called: {paper_dir.name} is done!")
            evaluation_result_dict[paper_dir.name + ".pdf"] = {
                "normalized_output": papers_proceed_dicts
            }
        else:
            print(f"Broken input: {paper_dir}")

    logger.info(
        f"No. of entries with dicts not lists: {no_of_dicts_instead_of_lists} n\ No. of tuples without result data: {triplets_without_result} \n"
        f"No. of tuples with results and not result: {no_of_triplets_with_results_instead_of_results}"
    )
    # save_dict_to_json(processed_dicts, os.path.join(output_dir, 'processed_tdmr_extraction_test_papers.json'))
    save_dict_to_json(
        evaluation_result_dict,
        os.path.join(
            output_dir,
            "processed_tdmr_extraction_test_papers_evaluation_valid_results.json",
        ),
    )
