import os
from io import StringIO
from pathlib import Path

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tqdm import tqdm

from src.logger import logger
from src.openai_client import (get_openai_model_response,
                               get_openai_model_structured_response)
from src.parsers.marker_parser import parse_pdf_with_marker
from src.parsers.parser import extract_pdf_sections_content
from src.utils import create_dir_if_not_exists, read_json, save_dict_to_json
from src.tdmr_extraction_utils.data_models import TdmrExtractionResponseSplit
from src.tdmr_extraction_utils.utils import (
    create_one_result_file,
    create_one_result_file_for_evaluation_purpose,
)

MODEL_NAME = "gpt-4-turbo"

from prompts.tdmr_extraction_without_model import (
    TDMR_EXTRACTION_PROMPT_07_02, TDMR_EXTRACTION_PROMPT_WITH_PARAGRAPHS,
    TDMR_EXTRACTION_PROMPT_07_02_no_format_instructions_prompt,
    TDMR_EXTRACTION_PROMPT_07_02_no_format_instructions_system_prompt,
    extract_table_id_prompt)


def main(extracted_triplet_path_dir, extracted_tables_dict_object, tdmr_extraction_dir):
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
                for (
                    table_object
                ) in (
                    extracted_tables_dict_object
                ):  # Path(extracted_tables_dir).iterdir():

                    csv_data = StringIO(table_object["data"])
                    table = pd.read_csv(csv_data)
                    table_caption = table_object["caption"]

                    # table_decision = decide_whether_table_contains_result_for_given_triplet(triplet, table, table_caption, TABLE_DECISION_PROMPT, {})
                    response = extract_result_from_given_table_for_triplet(
                        triplet, table, table_caption, structured_output=False
                    )
                    if response:
                        output_list.append(response)

    save_dict_to_json(
        output_list,
        os.path.join(
            tdmr_extraction_dir,
            f"{Path(extracted_triplet_path_dir).name}_tdmr_extraction.json",
        ),
    )


def main_extended_with_more_context(
    extracted_triplet_path_dir, extracted_tables_dict_object, tdmr_extraction_dir
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
                for (
                    table_object
                ) in (
                    extracted_tables_dict_object
                ):  # Path(extracted_tables_dir).iterdir():

                    csv_data = StringIO(table_object["data"])
                    table = pd.read_csv(csv_data)
                    table_caption = table_object["caption"]
                    if "context" in table_object:
                        context_list = table_object["context"]
                        for context_dict in context_list:

                            ### Version with deciding about proper context step
                            # table_decision = decide_whether_table_contains_result_for_given_triplet(triplet, table, table_caption, TABLE_DECISION_PROMPT_WITH_PARAGRAPHS, context_dict)
                            # if table_decision['decision'].lower() == 'true':
                            #     response = extract_result_from_given_table_for_triplet(triplet, table, table_caption, TDMR_EXTRACTION_PROMPT_07_02)
                            #     if response:
                            #         output_list.append(response)
                            # else:
                            #     logger.warning(f"The provided prompt could not yield satisfacotry results due to: {table_decision['explanation']}")
                            ### Ending the version with deciding of the correct context
                            response = extract_result_from_given_table_for_triplet_with_extended_context(
                                triplet,
                                table,
                                table_caption,
                                TDMR_EXTRACTION_PROMPT_WITH_PARAGRAPHS,
                                context_dict,
                            )
                            if response:
                                output_list.append(response)
                    else:
                        logger.warning(
                            "No addition context were retrieved, passing the prompt to simplified approach"
                        )
                        response = extract_result_from_given_table_for_triplet(
                            triplet, table, table_caption
                        )
                        if response:
                            output_list.append(response)

    save_dict_to_json(
        output_list,
        os.path.join(
            tdmr_extraction_dir,
            f"{Path(extracted_triplet_path_dir).name}_tdmr_extraction.json",
        ),
    )


def extract_result_from_given_table_for_triplet_with_extended_context(
    triplet: dict,
    table: pd.DataFrame,
    table_caption: str,
    prompt_template: str,
    context_dict: dict,
) -> dict:
    parser = JsonOutputParser(pydantc_object=None) # TODO fix this
    prompt = PromptTemplate(
        input_variables=[
            "triplet",
            "table",
            "table_caption",
            "paragraph_before",
            "paragraph_with_table",
            "paragraph_after",
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template=prompt_template,
    ).format(
        triplet=triplet,
        table=table.to_markdown(),
        table_caption=table_caption,
        paragraph_before=context_dict["before_paragraph"],
        paragraph_with_table=context_dict["current_paragraph"],
        paragraph_after=context_dict["next_paragraph"],
    )

    response = get_extract_from_openai(prompt, parser)

    return response


def extract_result_from_given_table_for_triplet(
    triplet: dict,
    table: pd.DataFrame,
    table_caption: str,
    structured_output: bool = True,
) -> dict:

    if structured_output:
        system_prompt = (
            TDMR_EXTRACTION_PROMPT_07_02_no_format_instructions_system_prompt
        )
        prompt_template = TDMR_EXTRACTION_PROMPT_07_02_no_format_instructions_prompt
        prompt = PromptTemplate(
            input_variables=["triplet", "table", "authors_model"],
            template=prompt_template,
        ).format(
            triplet=triplet, table=table.to_markdown(), table_caption=table_caption
        )
        response = get_openai_model_structured_response(
            prompt, TdmrExtractionResponseSplit, system_prompt=system_prompt
        )
        response = {
            "Task": response.task,
            "Dataset": response.dataset,
            "Metric": response.metric,
            "Result": response.result,
        }
    else:
        prompt_template = TDMR_EXTRACTION_PROMPT_07_02
        parser = JsonOutputParser(pydantc_object=TdmrExtractionResponseSplit)
        prompt = PromptTemplate(
            input_variables=["triplet", "table", "authors_model"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template=prompt_template,
        ).format(
            triplet=triplet, table=table.to_markdown(), table_caption=table_caption
        )
        response = get_extract_from_openai(prompt, parser)
    return response


def get_extract_from_openai(prompt: str, parser: JsonOutputParser):
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


def run_extending_context_experiment(
    test_paper_dir: str,
    extracted_tabels_origin_dir: str,
    extracted_triplet_dir_path: str,
    tdmr_extraction_dir: str,
):
    for paper_name in Path(test_paper_dir).iterdir():
        if paper_name.suffix == ".pdf":
            extended_table_object_file_path = (
                f"{test_paper_dir}/{paper_name.stem}/result.json"
            )

            if not Path(extended_table_object_file_path).exists():
                table_ids = set()
                marker_output_dir = f"{test_paper_dir}/{paper_name.stem}/marker_output"

                markdown_file_path = parse_pdf_with_marker(
                    str(paper_name), marker_output_dir
                )

                if not Path(markdown_file_path).exists():
                    logger.error(
                        f"for this paper we got a problem: {marker_output_dir}"
                    )
                    continue

                extracted_sections_content = extract_pdf_sections_content(
                    markdown_file_path
                )
                extacted_tabels_with_captions_fle_path = os.path.join(
                    extracted_tabels_origin_dir, paper_name.stem, "result_dict.json"
                )
                extracted_tables_with_captions = read_json(
                    Path(extacted_tabels_with_captions_fle_path)
                )

                for index, extracted_table in enumerate(extracted_tables_with_captions):
                    table_caption = extracted_table["caption"]

                    if table_caption:
                        print(f"The provided captions: {table_caption}")
                        table_id = extract_table_id_from_caption(
                            table_caption, extract_table_id_prompt
                        ).strip()
                        print(f"Extracted table_id {table_id}")

                        keys_list = list(extracted_sections_content.keys())

                        if table_id in table_ids:
                            logger.warning(f"Table id :{table_id} already in the set")
                            continue
                        else:
                            table_ids.add(table_id)
                            context_dicts = []
                            for i, (section_title, section_content) in enumerate(
                                extracted_sections_content.items()
                            ):
                                context_dict = {
                                    "before_paragraph": "",
                                    "current_paragraph": "",
                                    "next_paragraph": "",
                                }

                                if table_id.lower() in section_content.lower():
                                    print(section_title)
                                    if i == 0:
                                        context_dict["before_paragraph"] = ""
                                        context_dict["current_paragraph"] = (
                                            section_content
                                        )
                                        context_dict["next_paragraph"] = (
                                            extracted_sections_content[keys_list[i + 1]]
                                        )
                                    elif i == len(keys_list) - 1:
                                        context_dict["before_paragraph"] = (
                                            extracted_sections_content[keys_list[i - 1]]
                                        )
                                        context_dict["current_paragraph"] = (
                                            section_content
                                        )
                                        context_dict["next_paragraph"] = ""
                                    else:
                                        context_dict["before_paragraph"] = (
                                            extracted_sections_content[keys_list[i - 1]]
                                        )
                                        context_dict["current_paragraph"] = (
                                            section_content
                                        )
                                        context_dict["next_paragraph"] = (
                                            extracted_sections_content[keys_list[i + 1]]
                                        )
                                    context_dicts.append(context_dict)

                            extracted_tables_with_captions[index].update(
                                {"context": context_dicts}
                            )
                save_dict_to_json(
                    extracted_tables_with_captions, extended_table_object_file_path
                )
            else:
                extracted_tables_with_captions = read_json(
                    Path(extended_table_object_file_path)
                )
                triplets_dir_path = os.path.join(
                    extracted_triplet_dir_path, paper_name.stem
                )
                paper_name_output_path = Path(
                    f"{tdmr_extraction_dir}/{paper_name.stem}"
                )

                create_dir_if_not_exists(paper_name_output_path)
                main_extended_with_more_context(
                    triplets_dir_path,
                    extracted_tables_with_captions,
                    paper_name_output_path,
                )

    return extracted_tables_with_captions


def extract_table_id_from_caption(table_caption: str, prompt_template: str) -> str:
    formatted_prompt = prompt_template.format(table_caption=table_caption)
    response = get_openai_model_response(formatted_prompt)
    return response


if __name__ == "__main__":
    create_result_file: bool = True
    extracted_triplet_dir_path = (
        "triplets_normalization_all_leaderboards_papers_gpt-4-turbo"
    )
    tdmr_extraction_dir = f"tdmr_extraction/{MODEL_NAME}/with_updated_normalized_triplet_using_gpt-4-turbo"

    create_dir_if_not_exists(Path(tdmr_extraction_dir))
    path_with_tables_captions = "/Users/Michal/Dokumenty_mac/MasterThesis/docling_tryout/results"  # TODO replace this path to Yours
    # run_extending_context_experiment("extending_context_experiment/experiments_source_papers",
    #                                  path_with_tables_captions, extracted_triplet_dir_path, tdmr_extraction_dir)

    papers_with_extracted_tables = list(
        Path(path_with_tables_captions).iterdir()
    )  #### !!!! uncomment to run the whole experiment

    # papers_with_extracted_tables_just_names = ['1906.05012', '1811.09242', '1909.02188']
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
