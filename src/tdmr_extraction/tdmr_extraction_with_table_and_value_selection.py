import os
from io import StringIO
from pathlib import Path

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tqdm import tqdm

from prompts.tdmr_extraction_with_table_selection import (
    PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_PROMPT_TEMPLATE,
    PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_SYSTEM_PROMPT,
    PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_SYSTEM_PROMPT_GPT_4_TURBO)
from prompts.tdmr_extraction_without_model import (
    TDMR_EXTRACTION_PROMPT_07_02_no_format_instructions_prompt,
    TDMR_EXTRACTION_PROMPT_05_07_system_prompt_with_selecting_value,
    TDMR_EXTRACTION_PROMPT_05_07_system_prompt_with_selecting_value_GPT4_turbo)

from src.logger import logger
from src.openai_client import (get_openai_model_response,
                               get_llm_model_response)

from src.tdmr_extraction.tdmr_extraction_without_author_approach import (
    create_one_result_file, create_one_result_file_for_evaluation_purpose)
from src.tdmr_extraction_utils.data_models import (TdmrExtractionResponseSplit, PreparedTable,
                                                   TableDecisionResponse, PREPARED_DICT_INTO_STR_TEMPLATE)
from src.utils import create_dir_if_not_exists, read_json, save_dict_to_json
from src.const import BENCHMARK_TABLES

MODEL_NAME = "gpt-4-turbo"

SYSTEM_PROMPT_RESULTS_EXTRACTION_MODEL_MAPPER = {"gpt-4-turbo": TDMR_EXTRACTION_PROMPT_05_07_system_prompt_with_selecting_value_GPT4_turbo,
                                                 "openai-gpt-oss-120b": TDMR_EXTRACTION_PROMPT_05_07_system_prompt_with_selecting_value}

SYSTEM_PROMPT_PICKING_UP_TABLEMODEL_MAPPER = {"gpt-4-turbo": PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_SYSTEM_PROMPT_GPT_4_TURBO,
                                              "openai-gpt-oss-120b": PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_SYSTEM_PROMPT}



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
            table=table.to_dict('records'), table_caption=table_caption, table_id=table_id
        )
        prepared_dicts.append(prepared_dict)
    return prepared_dicts


def pick_optimal_source_table_for_given_triplet(
    extracted_triplet: dict,
    prepared_dicts: list[PreparedTable],
    prompt_template: str,
    result_object: TableDecisionResponse,
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
        response = get_llm_model_response(prompt=prompt, pydantic_object_structured_output=result_object,
                                          system_prompt=system_prompt, model_name=MODEL_NAME)
        if response:
            if not isinstance(response, dict):
                response = response.model_dump()
    else:
        response = get_llm_model_response(
            prompt=prompt, pydantic_object_structured_output=None,
            system_prompt=system_prompt, model_name=MODEL_NAME
        )
    logger.info(f"Response for picking table: {response}")
    return response


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


def extract_result_from_given_table_for_triplet(
    triplet: dict,
    table_wit_additional_data: PreparedTable,
    system_prompt: str,
    prompt_template: str,
    result_object: TdmrExtractionResponseSplit,
    structured_output: bool = True,
) -> dict:
    if structured_output:
        prompt = PromptTemplate(
            input_variables=["triplet", "table", "table_caption"],
            template=prompt_template,
        ).format(
            triplet=triplet,
            table=table_wit_additional_data.dataframe.to_markdown(),
            table_caption=table_wit_additional_data.table_caption,
        )
        response = get_llm_model_response(
            prompt=prompt, pydantic_object_structured_output=result_object, system_prompt=system_prompt, model_name=MODEL_NAME
        )

        if isinstance(response, dict):
            try:
                response = TdmrExtractionResponseSplit(**response)
            except Exception as e:
                logger.error(f"The parsed response: {response} could not be transferred to a pydantic object due to: {str(e)}")
                response = None

        logger.info(f"Response for assigning value to the triplet: {response}")
        if response:
            response = {
                "Task": response.task,
                "Dataset": response.dataset,
                "Metric": response.metric,
                "Result": str(response.result),
            }
    else:
        parser = JsonOutputParser(pydantc_object=result_object)
        prompt_template = system_prompt + prompt_template
        prompt = PromptTemplate(
            input_variables=["triplet", "table", "table_caption"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template=prompt_template,
        ).format(
            triplet=triplet,
            table=table_wit_additional_data.table,
            table_caption=table_wit_additional_data.table_caption,
        )
        response = get_extract_from_openai(prompt, parser)
    return response


def get_extract_from_openai(prompt: str, parser: JsonOutputParser) -> dict:
    response = get_openai_model_response(prompt)
    try:
        response = parser.parse(response)
        print(response)

    except Exception as e:
        print(
            f"This response could not been parsed: {response} due to an exception: {e}"
        )
        response = {}

    return response


def main(
    extracted_triplet_path_dir,
    extracted_tables_dict_object: list[dict],
    tdmr_extraction_dir,
):
    output_list = []

    for triplet_path in Path(extracted_triplet_path_dir).iterdir():

        if str(triplet_path).endswith(".json"):
            try:
                triplet_set = read_json(triplet_path)
            except Exception as e:
                logger.error(
                    f"The provided triplet path is invalid: {str(e)}, continuing with next one"
                )
                continue

            for triplet in triplet_set:
                if triplet:
                    prepared_dicts_for_selecting_optimal_table = (
                        prepare_extracted_tables_for_experiment(
                            extracted_tables_dict_object=extracted_tables_dict_object
                        )
                    )
                    try:
                        system_prompt_per_model_picking_up_table = SYSTEM_PROMPT_PICKING_UP_TABLEMODEL_MAPPER[MODEL_NAME]
                        table_id_to_use = pick_optimal_source_table_for_given_triplet(
                            extracted_triplet=triplet,
                            prepared_dicts=prepared_dicts_for_selecting_optimal_table,
                            prompt_template=PICK_OPTIMAL_TABLE_WITHOUT_ADDITIONAL_CONTEXT_PROMPT_TEMPLATE,
                            result_object=TableDecisionResponse,
                            system_prompt=system_prompt_per_model_picking_up_table,
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
                    system_prompt_per_model_result_extraction = SYSTEM_PROMPT_RESULTS_EXTRACTION_MODEL_MAPPER[MODEL_NAME]
                    response = extract_result_from_given_table_for_triplet(
                        triplet,
                        table_dict,
                        structured_output=True,
                        system_prompt=system_prompt_per_model_result_extraction,
                        prompt_template=TDMR_EXTRACTION_PROMPT_07_02_no_format_instructions_prompt,
                        result_object=TdmrExtractionResponseSplit,
                    )
                    if response:
                        output_list.append(response)
        else:
            logger.warning(
                f"The provided path is not a valid triplet: {str(triplet_path)} It's not a JSON file!"
            )

    save_dict_to_json(
        output_list,
        os.path.join(
            tdmr_extraction_dir,
            f"{Path(extracted_triplet_path_dir).name}_tdmr_extraction.json",
        ),
    )


if __name__ == "__main__":
    create_result_file: bool = True
    extracted_triplet_dir_path = "triplets_normalization/gpt-4-turbo/from_chunk_approach_refined_prompt_12_08"
    tdmr_extraction_dir = (
        f"tdmr_extraction/{MODEL_NAME}/from_chunk_by_chunk_table_and_value_selection_13_08"
    )

    create_dir_if_not_exists(Path(tdmr_extraction_dir))
    path_with_tables_captions = (
        "/Users/Michal/Dokumenty_mac/MasterThesis/docling_tryout/results"
    )

    papers_with_extracted_tables = list(
        Path(path_with_tables_captions).iterdir()
    )  #### !!!! uncomment to run the whole experiment

    # papers_with_extracted_tables_just_names = BENCHMARK_TABLES
    # papers_with_extracted_tables = [Path(os.path.join(path_with_tables_captions, x)) for x in papers_with_extracted_tables_just_names]

    already_processed_files = [
        paper_path.name for paper_path in Path(tdmr_extraction_dir).iterdir()
    ]

    for paper_path in tqdm(papers_with_extracted_tables):
        logger.info(f"Analyzed file: {paper_path}")

        if Path(paper_path).is_dir():
            if paper_path.name in already_processed_files:
                print(f"File has been already processed: {paper_path.name}")
                continue
            paper_name_output_path = Path(f"{tdmr_extraction_dir}/{paper_path.name}")
            create_dir_if_not_exists(paper_name_output_path)

            extracted_triplet_path_dir = os.path.join(
                extracted_triplet_dir_path, paper_path.name
            )
            if Path(extracted_triplet_path_dir).exists():
                extracted_tables_with_captions = read_json(
                    Path(os.path.join(paper_path, "result_dict.json"))
                )
                main(
                    extracted_triplet_path_dir,
                    extracted_tables_with_captions,
                    paper_name_output_path,
                )
            else:
                logger.error(f"This path is broken: {extracted_triplet_path_dir}")

    if create_result_file:
        create_one_result_file(Path(tdmr_extraction_dir))
        create_one_result_file_for_evaluation_purpose(Path(tdmr_extraction_dir))
