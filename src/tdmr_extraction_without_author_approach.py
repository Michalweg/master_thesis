from pathlib import Path

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

TDMR_EXTRACTION_PROMPT = """
You will be given a triplet (an information piece which is constructed from the Dataset, Metric and Task) and a table with the results alongside its caption. 
Your task is to assign value to the extracted dataset, metric and task triplet based on the provided data in a table. 
Please output an updated dictionary with this result (so final dictionary consists of dataset, metric, task and extracted result)
Please note that table caption does not have it explicitly state the dataset for given triplet, in this case please assume
that dataset matches and extract result for the task, metric and model approach. 

Here is extracted triplet:
{triplet}

Here is the table with results:
{table} 

Here is the table caption:
{table_caption}

Here are some guidelines: 
1. If you cannot find the information about the result for specific triplet, output empty dict. 

{format_instructions}
"""

TDMR_EXTRACTION_PROMPT_07_02 = """
You will be given a triplet (an information piece which is constructed from the Dataset, Metric and Task) and a table with the results alongside its caption. 
Your task is to assign value to the extracted dataset, metric and task triplet based on the provided data in a table. 
Please output an updated dictionary with this result (so final dictionary consists of dataset, metric, task and extracted result)
Please note that table caption does not have it explicitly state the dataset for given triplet, in this case please assume
that dataset matches and extract result for the task, metric and model approach. 

Here is extracted triplet:
{triplet}

Here is the table with results:
{table} 

Here is the table caption:
{table_caption}

Here are some guidelines: 
1. If you cannot find the information about the result for specific triplet, output empty dict. 
2. If you will find multiple results for given triplet, please extract all results in the form of a list, so in the 'Result' dict will be a list containg all values. 

{format_instructions}
"""


def main(extracted_triplet_path_dir, extracted_tables_dict_object, tdmr_extraction_dir):
    output_list = []
    parser = JsonOutputParser(pydantc_object=TdmrExtractionResponse)

    for triplet_path in Path(extracted_triplet_path_dir).iterdir():

        if str(triplet_path).endswith(".json"):
            try:
                triplet_set = read_json(triplet_path)
            except:
                continue


            for triplet in triplet_set:
                for table_object in extracted_tables_dict_object:  # Path(extracted_tables_dir).iterdir():

                    csv_data = StringIO(table_object['data'])
                    table = pd.read_csv(csv_data)

                    table_caption = table_object['caption']
                    prompt = PromptTemplate(input_variables=['triplet', 'table', 'authors_model'],
                                            partial_variables={'format_instructions': parser.get_format_instructions()},
                                            template=TDMR_EXTRACTION_PROMPT_07_02).format(triplet=triplet,
                                                                                    table=table.to_markdown(),
                                                                                    table_caption=table_caption)

                    response = get_openai_model_response(prompt)
                    try:
                        response = parser.parse(response)
                        print(response)
                        if response:
                            output_list.append(response)

                    except Exception as e:
                        print(f"This response could not been parsed: {response}")

    save_dict_to_json(output_list, os.path.join(tdmr_extraction_dir, f'{Path(extracted_triplet_path_dir).name}_tdmr_extraction.json'))


def create_one_result_file(output_dir: Path):
    processed_dicts = []
    all_unique_papersInitial = set()
    for index, paper_dir in enumerate(list(output_dir.iterdir())):
        print(index)
        paper_result_file_path = os.path.join(paper_dir, paper_dir.name + '_tdmr_extraction.json')
        paper_result_json = read_json(Path(paper_result_file_path))
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
                    # elif isinstance(result_dict[result_keyword], list):
                    #     new_dicts = []
                    #     for item in result_dict[result_keyword]:
                    #         if isinstance(item, dict):
                    #             for k, v in item.items():
                    #                 new_result_dict = stay_dict.copy()
                    #                 if k == 'Result' or k == 'Results':
                    #                     new_result_dict.update({k: v})
                    #                     new_dicts.append(new_result_dict)


                    else:
                        processed_dicts.append(result_dict)
                else:
                    processed_dicts.append(result_dict)
            else:
                print(result_dict)
    save_dict_to_json(processed_dicts, os.path.join(output_dir, 'processed_tdmr_extraction.json'))



if __name__ == "__main__":
    tdmr_extraction_dir = f"tdmr_extraction/{MODEL_NAME}/with_captions_updated_tables_07_02_new_table_representation"
    create_dir_if_not_exists(Path(tdmr_extraction_dir))
    # create_one_result_file(Path(tdmr_extraction_dir))

    extracted_triplet_dir_path = f"triplets_extraction/from_entire_document_refined_prompt_gpt_4o"

    path_with_tables_captions = '/Users/Michal/Dokumenty_mac/MasterThesis/docling_tryout/results'
    # papers_with_extracted_tables = list(Path(path_with_tables_captions).iterdir()) -> # TODO uncomment these lines as well!
    papers_with_extracted_tables_just_names = ['1906.05012', '1811.09242', '1909.02188']
    papers_with_extracted_tables = [Path(os.path.join(path_with_tables_captions, x)) for x in papers_with_extracted_tables_just_names]
    already_processed_files = [paper_path.name for paper_path in Path(tdmr_extraction_dir).iterdir()]


    for i, paper_path in enumerate(papers_with_extracted_tables):
        if '1909' not in paper_path.name:
            continue
        # TODO uncomment this line once you're done comparing files!
        # if paper_path.name in already_processed_files:
        #     print(f"File has been already processed: {paper_path.name}")
        #     continue
        paper_name_output_path = Path(f"{tdmr_extraction_dir}/{paper_path.name}")
        create_dir_if_not_exists(paper_name_output_path)

        extracted_triplet_path_dir = os.path.join(extracted_triplet_dir_path, paper_path.name)

        extracted_tables_with_captions = read_json(Path(os.path.join(paper_path, "result_dict.json")))
        main(extracted_triplet_path_dir, extracted_tables_with_captions, paper_name_output_path)
        # if i == 20:
        #     break