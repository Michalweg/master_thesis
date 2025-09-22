from pathlib import Path

from src.utils import create_dir_if_not_exists, save_dict_to_json


# Load documents
import os
from io import StringIO

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.logger import logger
from src.openai_client import get_openai_model_response, get_llm_model_response
from src.utils import read_json
from src.tdmr_extraction_utils.data_models import TdmrExtractionResponseWithModel
from prompts.tdmr_extracton_with_model_name import TDMR_EXTRACTION_SYSTEM_PROMPT, TDMR_EXTRACTION_USER_PROMPT

MODEL_NAME = "openai-gpt-oss-120b"



def main(
    extracted_triplet_path_dir: str,
    all_extracted_author_approach: set,
    extracted_tables_summary_file_path: str,
    tdmr_extraction_output_dir_path: str,
    pydantic_object: type[BaseModel],
    user_prompt: str,
    system_prompt: str,
    model_name: str
):
    output_list = []
    extracted_tables_and_captions = read_json(Path(extracted_tables_summary_file_path))

    for author_approach in all_extracted_author_approach:
        for triplet_path in Path(extracted_triplet_path_dir).iterdir():
            if str(triplet_path).endswith(".json"):
                try:
                    triplet_set = read_json(triplet_path)
                except:
                    continue

                for triplet in triplet_set:
                    for table_object in extracted_tables_and_captions:

                        csv_data = StringIO(table_object["data"])
                        table = pd.read_csv(csv_data)
                        table_caption = table_object["caption"]

                        prompt = PromptTemplate(
                            input_variables=[
                                "triplet",
                                "table",
                                "authors_model",
                                "table_caption",
                            ],
                            template=user_prompt,
                        ).format(
                            triplet=triplet,
                            table=table.to_markdown(),
                            authors_model=author_approach,
                            table_caption=table_caption,
                        )
                        response =  get_llm_model_response(prompt, model_name=model_name, system_prompt=system_prompt, pydantic_object_structured_output=pydantic_object)
                        logger.info(f"Raw response: {response}")
                        if response:
                            response = response if isinstance(response, dict) else response.model_dump()
                            if response['tdmr_output']:
                                if "Result" in response:
                                    if response["Result"]:
                                        response.update({"Model": author_approach})
                                        output_list.append(response)

    save_dict_to_json(
        output_list,
        os.path.join(
            tdmr_extraction_output_dir_path,
            f"{Path(extracted_triplet_path_dir).name}_tdmr_extraction.json",
        ),
    )


def define_all_unique_extracted_approaches_names(
    list_of_all_extracted_approaches_per_section: list[dict],
) -> set:
    all_extracted_authors_approach = set()
    for extracted_approach_dict in list_of_all_extracted_approaches_per_section:
        for section_name in extracted_approach_dict:
            list_of_extracted_approaches_in_section = extracted_approach_dict[
                section_name
            ]
            if list_of_extracted_approaches_in_section:
                for extracted_approach in list_of_extracted_approaches_in_section:
                    all_extracted_authors_approach.add(extracted_approach)

    return all_extracted_authors_approach


def create_one_result_file(output_dir: Path):
    processed_dicts = []
    all_unique_papersInitial = set()
    for index, paper_dir in enumerate(list(output_dir.iterdir())):
        paper_result_file_path = paper_dir
        paper_result_json = read_json(Path(str(paper_result_file_path)))
        for result_dict in paper_result_json:
            if isinstance(result_dict, dict):
                result_dict.update({"PaperName": paper_dir.stem.split("_")[0]})
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
                        if result_dict[result_keyword]:
                            processed_dicts.append(result_dict)
                else:
                    processed_dicts.append(result_dict)
            else:
                print(result_dict)
    save_dict_to_json(
        processed_dicts,
        os.path.join(
            Path(output_dir).parent, "tdmr_extraction_with_author_data_combined.json"
        ),
    )


if __name__ == "__main__":
    ### For each analyzed paper in the dir of the paper should be inputs and outputs that led to the given result
    # Specifying a dir with papers to analyze
    author_model_approach_experiment_dir_path = (
        "author_model_extraction/openai-gpt-oss-120b/from_each_section_22_09"
    )
    papers_dir = "custom_dataset_papers/CronQuestions"  # os.path.join(author_model_approach_experiment_dir_path, "papers")
    papers_to_analyze: list = [f for f in Path(papers_dir).iterdir() if f.suffix == ".pdf"]

    # Specifying specific dir where all triplets for all papers are defined
    extracted_normalized_triplet_dir_path = "triplets_normalization/openai-gpt-oss-120b/chunk_focus_approach/20_09_update"

    # Specifying a dir with tables and captions
    path_with_tables_captions = (
        "/Users/Michal/Dokumenty_mac/MasterThesis/docling_tryout/custom_dataset_papers/CronQuestions"
    )

    # Setting up output dir
    tdmr_extraction_output_dir = os.path.join(
        "author_model_extraction_results",
        f"{MODEL_NAME}/with_author_model_approach_20_09_2025",
    )
    create_dir_if_not_exists(Path(tdmr_extraction_output_dir))

    already_processed_file = [f.name for f in Path(tdmr_extraction_output_dir).iterdir() if f.suffix == ".pdf"]
    create_dir_if_not_exists(Path(tdmr_extraction_output_dir))

    for paper_path in papers_to_analyze:

        if paper_path.stem in already_processed_file:
            logger.info(f"File: {paper_path} already processed ...")
            continue

        logger.info(f"Processing: {paper_path}")

        # Extracting all unique values for extracted approaches
        summary_of_approach_model_algorithm_file_path_per_section = os.path.join(author_model_approach_experiment_dir_path, paper_path.stem, "author_model_approaches.json")
        list_of_all_extracted_approaches_per_section = read_json(Path(summary_of_approach_model_algorithm_file_path_per_section))
        list_of_all_extracted_approaches_per_section = [list_of_all_extracted_approaches_per_section[-1]] if "most_common_model_name" in list_of_all_extracted_approaches_per_section[-1] else list_of_all_extracted_approaches_per_section
        all_extracted_authors_approach = define_all_unique_extracted_approaches_names(list_of_all_extracted_approaches_per_section) if len(list_of_all_extracted_approaches_per_section) > 1 else [list_of_all_extracted_approaches_per_section[0]['most_common_model_name']]

        if all_extracted_authors_approach:
            # Specifying path to all extracted triplets
            extracted_triplet_path_dir = os.path.join(extracted_normalized_triplet_dir_path, paper_path.stem)
            # Specifying path to all extracted tables with captions
            extracted_tables_file_path = os.path.join(path_with_tables_captions, paper_path.stem, "result_dict.json")

            main(extracted_triplet_path_dir, all_extracted_authors_approach, extracted_tables_file_path,
                 tdmr_extraction_output_dir, user_prompt=TDMR_EXTRACTION_USER_PROMPT,
                 system_prompt=TDMR_EXTRACTION_SYSTEM_PROMPT, pydantic_object=TdmrExtractionResponseWithModel, model_name=MODEL_NAME)
        else:
            logger.warning(f"No extracted results for paper {paper_path.stem}")
            continue

    create_one_result_file(Path(tdmr_extraction_output_dir))
