from pathlib import Path

from src.utils import create_dir_if_not_exists, save_dict_to_json

MODEL_NAME = "gpt-4o"

# Load documents
import os

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from src.openai_client import get_openai_model_response
from src.utils import read_json

class TdmrExtractionResponse(BaseModel):
    tdmr_dict: dict = Field(description="An updated dictionary containing task, dataset, metric and metric value for an approach/model developed by authors of the paper.")

TDMR_EXTRACTION_PROMPT = """
You will be given a triplet (an information piece which is constructed from the Dataset, Metric and task), an information about
an approach/model that the authors of research papers worked on and a table with the results alongside its caption. Your task is to assign value to the extracted dataset, 
metric and task triplet for the model/approach designed by authors of the paper. Please output an updated dictionary with this result (so final dictionary consists 
of dataset, metric, task and extracted result)

Here is extracted triplet:
{triplet}

Here is the table with results:
{table} 

Here is the name about approach/model authors worked on:
{authors_model}

Here are some guidelines: 
1. If you cannot find the information about the result for specific triplet, output empty dict. 

{format_instructions}
"""

def main(extracted_triplet_path_dir, all_extracted_author_approach, extracted_tables_dir, tdmr_extraction_dir):
    output_list = []
    parser = JsonOutputParser(pydantc_object=TdmrExtractionResponse)
    for author_approach in all_extracted_author_approach:
        for triplet_path in Path(extracted_triplet_path_dir).iterdir():
            if str(triplet_path).endswith(".json"):
                try:
                    triplet_set = read_json(triplet_path)
                except:
                    continue
                for triplet in triplet_set:
                    for table_path in Path(extracted_tables_dir).iterdir():
                        table = pd.read_csv(table_path)
                        prompt = PromptTemplate(input_variables=['triplet', 'table', 'authors_model'],
                                                partial_variables={'format_instructions': parser.get_format_instructions()},
                                                template=TDMR_EXTRACTION_PROMPT).format(triplet=triplet,
                                                                                        table=table,
                                                                                        authors_model=author_approach)
                        response = get_openai_model_response(prompt)
                        print(response)
                        try:
                            response = parser.parse(response)
                            print(response)
                            output_list.append(response)
                            break
                        except:
                            print(response)
    save_dict_to_json(output_list, os.path.join(tdmr_extraction_dir, f'{Path(extracted_triplet_path_dir).name}_tdmr_extraction.json'))

if __name__ == "__main__":
    parsed_papers_without_table_content_dir = "parsing_experiments/15_12_2024_gpt-4o"
    parsed_papers_without_table_content = list(Path(parsed_papers_without_table_content_dir).iterdir())
    tdmr_extraction_dir = f"tdmr_extraction/{MODEL_NAME}"
    extracted_authors_approach_dir_path = f"author_model_extraction/from_each_section_{MODEL_NAME}_no_vector_db"
    extracted_triplet_dir_path = f"triplets_extraction/from_each_section_with_table_{MODEL_NAME}"
    create_dir_if_not_exists(Path(tdmr_extraction_dir))


    for paper_path in parsed_papers_without_table_content:
        paper_name_output_path = Path(f"{tdmr_extraction_dir}/{paper_path.name}")
        create_dir_if_not_exists(paper_name_output_path)

        extracted_approach_dir_path = os.path.join(extracted_authors_approach_dir_path,
                                                    paper_path.name)
        all_extracted_authors_approach = set()
        for section_path in Path(extracted_approach_dir_path).iterdir():
            try:
                author_approach = read_json(section_path)['extracted_model_approach_names']
                if author_approach:
                    for approach in author_approach:
                        all_extracted_authors_approach.add(approach)
            except:
                continue

        extracted_triplet_path_dir = os.path.join(extracted_triplet_dir_path, paper_path.name)
        extracted_tables_dir = os.path.join(parsed_papers_without_table_content_dir, paper_path.name, 'manual_extracted_tables')
        if not Path(extracted_tables_dir).exists():
            continue

        main(extracted_triplet_path_dir, all_extracted_authors_approach, extracted_tables_dir, tdmr_extraction_dir,)