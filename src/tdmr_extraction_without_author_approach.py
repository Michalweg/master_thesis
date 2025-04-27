from pathlib import Path

from src.logger import logger
from src.utils import create_dir_if_not_exists, save_dict_to_json

MODEL_NAME = "gpt-4o"
import os
from io import StringIO

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.openai_client import get_openai_model_response
from src.utils import read_json


class TdmrExtractionResponse(BaseModel):
    tdmr_dict: dict = Field(description="An updated dictionary containing task, dataset, metric and metric value.")

class TableDecisionResponse(BaseModel):
    explanation: str = Field(description="An explanation of the decision of choosing the table to extract result for provided triplet")
    decision: bool = Field(description="Whether the table contains result for given triplet")

from prompts.tdmr_extraction_without_model import (
    TDMR_EXTRACTION_PROMPT_05_04, TDMR_EXTRACTION_PROMPT_07_02, TABLE_DECISION_PROMPT)


def main(extracted_triplet_path_dir, extracted_tables_dict_object, tdmr_extraction_dir):
    output_list = []

    for triplet_path in Path(extracted_triplet_path_dir).iterdir():

        if str(triplet_path).endswith(".json"):

            try:
                triplet_set = read_json(triplet_path)
            except Exception as e:
                logger.error(f"The provided triplet path is invalid: {str(e)}, continuing with next one")
                continue

            for triplet in triplet_set:
                for table_object in extracted_tables_dict_object:  # Path(extracted_tables_dir).iterdir():

                    csv_data = StringIO(table_object['data'])
                    table = pd.read_csv(csv_data)
                    table_caption = table_object['caption']

                    table_decision = decide_whether_table_contains_result_for_given_triplet(triplet, table, table_caption, TABLE_DECISION_PROMPT)
                    if table_decision['decision']:
                        response = extract_result_from_given_table_for_triplet(triplet, table, table_caption, TDMR_EXTRACTION_PROMPT_07_02)
                        if response:
                            output_list.append(response)



    save_dict_to_json(output_list, os.path.join(tdmr_extraction_dir, f'{Path(extracted_triplet_path_dir).name}_tdmr_extraction.json'))


def extract_result_from_given_table_for_triplet(triplet: dict, table: pd.DataFrame, table_caption: str,
                                                prompt_template: str) -> dict:
    parser = JsonOutputParser(pydantc_object=TdmrExtractionResponse)
    prompt = PromptTemplate(input_variables=['triplet', 'table', 'authors_model'],
                            partial_variables={'format_instructions': parser.get_format_instructions()},
                            template=prompt_template).format(triplet=triplet,
                                                                          table=table.to_markdown(),
                                                                          table_caption=table_caption)

    response = get_extract_from_openai(prompt, parser)

    return response


def get_extract_from_openai(prompt: str, parser: JsonOutputParser):
    response = get_openai_model_response(prompt)
    try:
        response = parser.parse(response)
        print(response)

    except Exception as e:
        print(f"This response could not been parsed: {response} due to an exception: {e}")
        response = {}

    return response


def decide_whether_table_contains_result_for_given_triplet(triplet: dict, table: pd.DataFrame, table_caption: str,
                                                           prompt_template: str) -> bool:
    parser = JsonOutputParser(pydantc_object=TableDecisionResponse)
    prompt = PromptTemplate(input_variables=['triplet', 'table', 'authors_model'],
                            partial_variables={'format_instructions': parser.get_format_instructions()},
                            template=prompt_template).format(triplet=triplet,
                                                             table=table.to_markdown(),
                                                             table_caption=table_caption)

    response = get_extract_from_openai(prompt, parser)

    return response


def create_one_result_file(output_dir: Path):
    processed_dicts = []
    all_unique_papersInitial = set()
    for index, paper_dir in enumerate(list(output_dir.iterdir())):
        paper_result_file_path = os.path.join(paper_dir, paper_dir.name + '_tdmr_extraction.json')
        paper_result_json = read_json(Path(str(paper_result_file_path)))
        for result_dict in paper_result_json:
            if isinstance(result_dict, dict):
                result_dict.update({"PaperName": paper_dir.name})
                all_unique_papersInitial.add(paper_dir.name)
                print(paper_dir.name)
                result_keyword = 'Result' if 'Result' in result_dict else 'Results'
                if result_keyword in result_dict:
                    stay_dict = {k: v for k, v in result_dict.items() if k != result_keyword}
                    if isinstance(result_dict[result_keyword], dict):
                        new_dicts = []
                        for k, v in result_dict[result_keyword].items():
                            new_result_dict = stay_dict.copy()
                            new_result_dict.update({'Result': v})
                            new_dicts.append(new_result_dict)
                        processed_dicts.extend(new_dicts)

                    else:
                        processed_dicts.append(result_dict)
                else:
                    processed_dicts.append(result_dict)
            else:
                print(result_dict)
    save_dict_to_json(processed_dicts, os.path.join(output_dir, 'processed_tdmr_extraction_test_papers.json'))



if __name__ == "__main__":
    create_result_file: bool = True
    tdmr_extraction_dir = f"tdmr_extraction/{MODEL_NAME}/with_captions_updated_tables_19_04_new_table_representation"
    create_dir_if_not_exists(Path(tdmr_extraction_dir))

    extracted_triplet_dir_path = "triplets_normalization"

    path_with_tables_captions = '/Users/Michal/Dokumenty_mac/MasterThesis/docling_tryout/results'
    # papers_with_extracted_tables = list(Path(path_with_tables_captions).iterdir()) -> # TODO uncomment to run the whole experiment

    papers_with_extracted_tables_just_names = ['1906.05012', '1811.09242', '1909.02188']
    papers_with_extracted_tables = [Path(os.path.join(path_with_tables_captions, x)) for x in papers_with_extracted_tables_just_names]
    already_processed_files = [paper_path.name for paper_path in Path(tdmr_extraction_dir).iterdir()]

    test_papers = ['1811.09242', '1906.05012', '1909.02188']
    for i, paper_path in enumerate(papers_with_extracted_tables):
        if  paper_path.name not in test_papers:
            continue
        logger.info(f"Analyzed file: {paper_path}")
        # if paper_path.name in already_processed_files:
        #     print(f"File has been already processed: {paper_path.name}")
        #     continue
        paper_name_output_path = Path(f"{tdmr_extraction_dir}/{paper_path.name}")
        create_dir_if_not_exists(paper_name_output_path)

        extracted_triplet_path_dir = os.path.join(extracted_triplet_dir_path, paper_path.name)

        extracted_tables_with_captions = read_json(Path(os.path.join(paper_path, "result_dict.json")))
        main(extracted_triplet_path_dir, extracted_tables_with_captions, paper_name_output_path)

    if create_result_file:
        create_one_result_file(Path(tdmr_extraction_dir))