from pathlib import Path

from src.utils import create_dir_if_not_exists, save_dict_to_json

MODEL_NAME = "gpt-4o"

# Load documents
import os
from io import StringIO

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.logger import logger
from src.openai_client import get_openai_model_response
from src.utils import read_json


class TdmrExtractionResponse(BaseModel):
    tdmr_dict: dict = Field(description="An updated dictionary containing task, dataset, metric and metric value for an approach/model developed by authors of the paper.")

TDMR_EXTRACTION_PROMPT = """
You will be given a triplet (an information piece which is constructed from the Dataset, Metric and task), an information about
an approach/model that the authors of research papers worked on and a table with the results alongside its caption. Your task is to assign value to the extracted dataset, 
metric and task triplet for the model/approach designed by authors of the paper. Please output an updated dictionary with this result (so final dictionary consists 
of dataset, metric, task and extracted result)
Please note that table caption does not have it explicitly state the dataset for given triplet, in this case please assume
that dataset matches and extract result for the task, metric and model approach. 

Here is extracted triplet:
{triplet}

Here is the table with results:
{table} 

Here is the table caption:
{table_caption}

Here is the name about approach/model authors worked on:
{authors_model}

Here are some guidelines: 
1. If you cannot find the information about the result for specific triplet, output empty dict. 
2. If you will find multiple results for given triplet, please extract all results in the form of a list, so in the 'Result' dict will be a list containing all values. If no results can be found, output an empty dict!
3. **DO NOT CREATE** a dictionary with keys such as "models" and the corresponding results as a value for Results section, you **SHOULD** put a value in "Result" section 
(or list of values if multiple were found) 

{format_instructions}
"""

def main(extracted_triplet_path_dir: str, all_extracted_author_approach: set, extracted_tables_summary_file_path: str, tdmr_extraction_output_dir_path: str):
    output_list = []
    parser = JsonOutputParser(pydantc_object=TdmrExtractionResponse)

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

                        csv_data = StringIO(table_object['data'])
                        table = pd.read_csv(csv_data)
                        table_caption = table_object['caption']

                        prompt = PromptTemplate(input_variables=['triplet', 'table', 'authors_model', 'table_caption'],
                                                partial_variables={'format_instructions': parser.get_format_instructions()},
                                                template=TDMR_EXTRACTION_PROMPT).format(triplet=triplet,
                                                                                        table=table,
                                                                                        authors_model=author_approach,
                                                                                        table_caption=table_caption)
                        response = get_openai_model_response(prompt)
                        logger.info(f"Raw response: {response}")
                        try:
                            response = parser.parse(response)
                            print(response)
                            if response:
                                if "Result" in response:
                                    if response["Result"]:
                                        response.update({"Model": author_approach})
                                        output_list.append(response)
                        except:
                            print(response)
    save_dict_to_json(output_list, os.path.join(tdmr_extraction_output_dir_path, f'{Path(extracted_triplet_path_dir).name}_tdmr_extraction.json'))


def define_all_unique_extracted_approaches_names(list_of_all_extracted_approaches_per_section: list[dict]) -> set:
    all_extracted_authors_approach = set()
    for extracted_approach_dict in list_of_all_extracted_approaches_per_section:
        for section_name in extracted_approach_dict:
            list_of_extracted_approaches_in_section = extracted_approach_dict[section_name]
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
                        if result_dict[result_keyword]:
                            processed_dicts.append(result_dict)
                else:
                    processed_dicts.append(result_dict)
            else:
                print(result_dict)
    save_dict_to_json(processed_dicts, os.path.join(Path(output_dir).parent, 'tdmr_extraction_with_author_data_combined.json'))

if __name__ == "__main__":
    ### For each analyzed paper in the dir of the paper should be inputs and outputs that led to the given result
    # Specifying a dir with papers to analyze
    author_model_approach_experiment_dir_path = "extending_results_extracton_with_author_approach"
    papers_to_extract_dir_path = os.path.join(author_model_approach_experiment_dir_path, "papers")
    list_of_posix_paths_for_papers_to_analyze = list(Path(papers_to_extract_dir_path).iterdir())

    # Specifying specific dir where all triplets for all papers are defined
    extracted_normalized_triplet_dir_path = "triplets_normalization"

    # Specifying a dir with tables and captions
    path_with_tables_captions = '/Users/Michal/Dokumenty_mac/MasterThesis/docling_tryout/results'


    # Setting up output dir
    tdmr_extraction_output_dir = os.path.join(author_model_approach_experiment_dir_path, f"{MODEL_NAME}/with_author_model_approach")
    create_one_result_file(Path(tdmr_extraction_output_dir))
    # create_dir_if_not_exists(Path(tdmr_extraction_output_dir))
    #
    # for paper_path in list_of_posix_paths_for_papers_to_analyze:
    #     # Extracting all unique values for extracted approaches
    #     summary_of_approach_model_algorithm_file_path_per_section = os.path.join(author_model_approach_experiment_dir_path, paper_path.stem, "author_model_approaches.json")
    #     list_of_all_extracted_approaches_per_section = read_json(Path(summary_of_approach_model_algorithm_file_path_per_section))
    #     all_extracted_authors_approach = define_all_unique_extracted_approaches_names(list_of_all_extracted_approaches_per_section)
    #
    #     if all_extracted_authors_approach:
    #         # Specifying path to all extracted triplets
    #         extracted_triplet_path_dir = os.path.join(extracted_normalized_triplet_dir_path, paper_path.stem)
    #         # Specifying path to all extracted tables with captions
    #         extracted_tables_file_path = os.path.join(path_with_tables_captions, paper_path.stem, "result_dict.json")
    #
    #         main(extracted_triplet_path_dir, all_extracted_authors_approach, extracted_tables_file_path, tdmr_extraction_output_dir)
    #     else:
    #         logger.warning(f"No extracted results for paper {paper_path.stem}")