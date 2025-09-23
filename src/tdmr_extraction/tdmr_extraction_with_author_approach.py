import os
from pathlib import Path

from langchain.prompts import PromptTemplate
from pydantic import BaseModel

from prompts.tdmr_extraction_with_table_selection import (
    PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_PROMPT_TEMPLATE,
    PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_SYSTEM_PROMPT,
    PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_SYSTEM_PROMPT_GPT_4_TURBO,
)
from prompts.tdmr_extracton_with_model_name import (
    TDMR_EXTRACTION_SYSTEM_PROMPT,
    TDMR_EXTRACTION_USER_PROMPT,
)
from src.logger import logger
from src.openai_client import get_llm_model_response
from src.tdmr_extraction_utils.data_models import (
    TableDecisionResponse,
    TdmrExtractionResponseWithModel,
)
from src.tdmr_extraction_utils.utils import (
    pick_optimal_source_table_for_given_triplet,
    prepare_extracted_tables_for_experiment,
)
from src.utils import create_dir_if_not_exists, read_json, save_dict_to_json
from src.tdmr_extraction_utils.utils import (
    create_one_result_file,
    create_one_result_file_for_evaluation_purpose,
)

MODEL_NAME = "openai-gpt-oss-120b"

SYSTEM_PROMPT_PICKING_UP_TABLEMODEL_MAPPER = {
    "gpt-4-turbo": PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_SYSTEM_PROMPT_GPT_4_TURBO,
    "openai-gpt-oss-120b": PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_SYSTEM_PROMPT,
}


def main(
    extracted_triplet_path_dir: str,
    all_extracted_author_approach: set,
    extracted_tables_summary_file_path: str,
    tdmr_extraction_output_dir_path: Path,
    pydantic_object: type[BaseModel],
    user_prompt: str,
    system_prompt: str,
    model_name: str,
):
    output_list = []
    extracted_tables_and_captions = read_json(Path(extracted_tables_summary_file_path))

    for author_approach in all_extracted_author_approach:
        for triplet_path in Path(extracted_triplet_path_dir).iterdir():
            if str(triplet_path).endswith(".json"):

                try:
                    triplet_set = read_json(triplet_path)
                except FileNotFoundError as e:
                    logger.error(str(e))
                    continue

                for triplet in triplet_set:
                    if triplet:
                        prepared_dicts_for_selecting_optimal_table = prepare_extracted_tables_for_experiment(
                            extracted_tables_dict_object=extracted_tables_and_captions
                        )
                    else:
                        prepared_dicts_for_selecting_optimal_table = []

                    try:
                        system_prompt_per_model_picking_up_table = (
                            SYSTEM_PROMPT_PICKING_UP_TABLEMODEL_MAPPER[MODEL_NAME]
                        )
                        table_id_to_use = pick_optimal_source_table_for_given_triplet(
                            extracted_triplet=triplet,
                            prepared_dicts=prepared_dicts_for_selecting_optimal_table,
                            prompt_template=PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_PROMPT_TEMPLATE,
                            result_object=TableDecisionResponse,
                            system_prompt=system_prompt_per_model_picking_up_table,
                            model_name=MODEL_NAME,
                        )["table_id_to_extract_result_metric_from"]
                        table_dict = [
                            prepared_table_dict
                            for prepared_table_dict in prepared_dicts_for_selecting_optimal_table
                            if prepared_table_dict.table_id == int(table_id_to_use)
                        ][0]
                    except Exception as e:
                        logger.error(
                            f"The provided prepared table is invalid: {str(e)}, {triplet}, {paper_path}"
                        )
                        continue

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
                        table=table_dict.dataframe.to_markdown(),
                        authors_model=author_approach,
                        table_caption=table_dict.table_caption,
                    )
                    response = get_llm_model_response(
                        prompt,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        pydantic_object_structured_output=pydantic_object,
                    )
                    logger.info(f"Raw response: {response}")
                    if response:
                        response = (
                            response
                            if isinstance(response, dict)
                            else response.model_dump()
                        )
                        if response["tdmr_output"]:
                            if "Result" in response["tdmr_output"]:
                                output_list.append(response["tdmr_output"])

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


if __name__ == "__main__":
    ### For each analyzed paper in the dir of the paper should be inputs and outputs that led to the given result
    # Specifying a dir with papers to analyze
    author_model_approach_experiment_dir_path = (
        "author_model_extraction/openai-gpt-oss-120b/from_each_section_22_09"
    )
    papers_dir = "custom_dataset_papers/CronQuestions"  # os.path.join(author_model_approach_experiment_dir_path, "papers")
    papers_to_analyze: list = [
        f for f in Path(papers_dir).iterdir() if f.suffix == ".pdf"
    ]

    # Specifying specific dir where all triplets for all papers are defined
    extracted_normalized_triplet_dir_path = (
        "triplets_normalization/openai-gpt-oss-120b/chunk_focus_approach/20_09_update"
    )

    # Specifying a dir with tables and captions
    path_with_tables_captions = "/Users/Michal/Dokumenty_mac/MasterThesis/docling_tryout/custom_dataset_papers/CronQuestions"

    # Setting up output dir
    tdmr_extraction_output_dir = f"tdmr_extraction_with_author_approach/{MODEL_NAME}/from_chunk_by_chunk_table_and_value_selection_22_09"
    create_dir_if_not_exists(Path(tdmr_extraction_output_dir))

    already_processed_file = [
        f.name for f in Path(tdmr_extraction_output_dir).iterdir() if f.suffix == ".pdf"
    ]
    create_dir_if_not_exists(Path(tdmr_extraction_output_dir))

    for paper_path in papers_to_analyze:

        if paper_path.stem in already_processed_file:
            logger.info(f"File: {paper_path} already processed ...")
            continue

        logger.info(f"Processing: {paper_path}")

        paper_name_output_path = Path(f"{tdmr_extraction_output_dir}/{paper_path.stem}")
        create_dir_if_not_exists(paper_name_output_path)

        # Extracting all unique values for extracted approaches
        summary_of_approach_model_algorithm_file_path_per_section = os.path.join(
            author_model_approach_experiment_dir_path,
            paper_path.stem,
            "author_model_approaches.json",
        )
        list_of_all_extracted_approaches_per_section = read_json(
            Path(summary_of_approach_model_algorithm_file_path_per_section)
        )
        list_of_all_extracted_approaches_per_section = (
            [list_of_all_extracted_approaches_per_section[-1]]
            if "most_common_model_name"
            in list_of_all_extracted_approaches_per_section[-1]
            else list_of_all_extracted_approaches_per_section
        )
        all_extracted_authors_approach = (
            define_all_unique_extracted_approaches_names(
                list_of_all_extracted_approaches_per_section
            )
            if len(list_of_all_extracted_approaches_per_section) > 1
            else [
                list_of_all_extracted_approaches_per_section[0][
                    "most_common_model_name"
                ]
            ]
        )

        if all_extracted_authors_approach:
            # Specifying path to all extracted triplets
            extracted_triplet_path_dir = os.path.join(
                extracted_normalized_triplet_dir_path, paper_path.stem
            )
            # Specifying path to all extracted tables with captions
            extracted_tables_file_path = os.path.join(
                path_with_tables_captions, paper_path.stem, "result_dict.json"
            )

            main(
                extracted_triplet_path_dir,
                all_extracted_authors_approach,
                extracted_tables_file_path,
                paper_name_output_path,
                user_prompt=TDMR_EXTRACTION_USER_PROMPT,
                system_prompt=TDMR_EXTRACTION_SYSTEM_PROMPT,
                pydantic_object=TdmrExtractionResponseWithModel,
                model_name=MODEL_NAME,
            )
        else:
            logger.warning(f"No extracted results for paper {paper_path.stem}")
            continue

    create_one_result_file(Path(tdmr_extraction_output_dir))
