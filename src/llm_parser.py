import json
import os
import random
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from src.parser_prompts import (PARSER_PROMPT_TEMPLATE,
                            phi3_medium_prompt_template_specific_csv)
from src.logger import logger
from src.utils import read_markdown_file_content, save_dict_to_json, save_str_as_markdown
from datetime import datetime
from tqdm import tqdm

class ModelResponse(BaseModel):
    tables_dict: dict = Field(description="Python dictionary containing key in form of 'table_1', 'table_2, etc. and values"
                                          "as strings containing csv file for each extracted table")

def parse_markdown_with_mistral(markdown_file_path: str | Path, output_dir: str | Path,
                                url: str = 'http://localhost:11434/api/generate', model_name: str = 'mistral', ):
    logger.info(f"Parsing Markdown with {model_name} ...")
    file_content = read_markdown_file_content(markdown_file_path)

    output_parser = JsonOutputParser(pydantic_object=ModelResponse)

    prompt = PromptTemplate(template=phi3_medium_prompt_template_specific_csv).format(markdown_file_content=file_content)
                            # partial_variables={"format_instructions": output_parser.get_format_instructions()})

    # prompt = PARSER_PROMPT_TEMPLATE.replace("{markdown_file_content}", file_content)
    payload = \
        {
          "model": model_name,
          "prompt": prompt,
          "stream": False,
          "temperature": 0.0
        }
    r = requests.post(url, json=payload)

    r = json.loads(r.content)['response']
    if r:
        correct_output_format = 0
        try:
            parsed = output_parser.parse(r)
            current_datetime = datetime.now()

            # Format the datetime to display only up to the minute
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
            os.makedirs(os.path.join(str(output_dir), Path(markdown_file_path).stem), exist_ok=True)
            save_dict_to_json(parsed, os.path.join(str(output_dir), Path(markdown_file_path).stem, f"{formatted_datetime}" + '.json'))
        except Exception as e:
            logger.warning(f"Saving extracted response failed due to an error: {e}")
        try:
            parsed = output_parser.parse(r)
            print(parsed)
            for table_id in parsed:
                try:
                    df = pd.read_csv(StringIO(parsed[table_id]), sep=',')
                    print(df)
                    correct_output_format += 1
                except Exception as e:
                    print(f"There is an error with parsing output {e}")
        except Exception as e:
            pass
    else:
        logger.info(f"In the provided markdown file there was no tables")
    logger.info(f"Parsing Markdown with {model_name} done")

def parse_markdown_sections(sections_dict: dict, output_dir: str | Path, model_name: str):
    for section in sections_dict.keys():
        section_content = sections_dict[section]
        save_str_as_markdown("temp_markdown.md", section_content)
        parse_markdown_with_mistral("temp_markdown.md", output_dir, model_name=model_name)

if __name__ == '__main__':
    model_name = 'phi3:medium'
    markdown_file_path = "../table_1.md"
    llm_parsed_tables_dir = Path("../llm_parsed_tables")

    for markdown_file in [f"../table_{i}.md" for i in range(1,4)]:
        for _ in tqdm(range(5)):
            parse_markdown_with_mistral(markdown_file_path=markdown_file,
                                        output_dir=llm_parsed_tables_dir, model_name=model_name)