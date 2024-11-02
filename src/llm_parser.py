import json
import os
from pathlib import Path

import requests
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from src.parser_prompts import gpt_4_prompt
from src.logger import logger
from src.utils import read_markdown_file_content, save_dict_to_json, save_str_as_markdown, save_str_as_txt_file, create_dir_if_not_exists, read_json
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

    prompt = PromptTemplate(template=gpt_4_prompt).format(markdown_file_content=file_content)
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

    # Parsing and saving LLM response
    r = json.loads(r.content)['response']

    if r:
        correct_output_format = 0
        try:
            parsed = output_parser.parse(r)
            # current_datetime = datetime.now()
            #
            # # Format the datetime to display only up-to-the-minute
            # formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
            # os.makedirs(os.path.join(str(output_dir), Path(markdown_file_path).stem), exist_ok=True)
            save_dict_to_json(parsed, os.path.join(str(output_dir), Path(markdown_file_path).stem + f"len_{len(file_content)}" + '.json'))
        except Exception as e:
            logger.warning(f"Saving extracted response failed due to an error: {e}")
            save_str_as_txt_file(txt_file_path=os.path.join(str(output_dir), Path(markdown_file_path).stem + f"len_{len(file_content)}" + '.txt'), str_content=r)


        # try:
        #     parsed = output_parser.parse(r)
        #     for table_id in parsed:
        #         try:
        #             df = pd.read_csv(StringIO(parsed[table_id]), sep=',')
        #             correct_output_format += 1
        #         except Exception as e:
        #             print(f"There is an error with parsing output {e}")
        # except Exception as e:
        #     pass
    else:
        logger.info(f"The model didn't generate any response")
        save_str_as_txt_file(txt_file_path=os.path.join(str(output_dir), Path(markdown_file_path).stem + f"len_{len(file_content)}" + '.txt'), str_content=r)
    logger.info(f"Parsing Markdown with {model_name} done")

def parse_markdown_sections(sections_dict: dict, output_dir: str | Path, model_name: str):
    for section in sections_dict.keys():
        section_content = sections_dict[section]
        save_str_as_markdown(f"{section}.md", section_content)
        parse_markdown_with_mistral(f"{section}.md", output_dir, model_name=model_name)
        os.remove(f"{section}.md")

if __name__ == '__main__':

    model_name = 'llama3.1'
    markdown_file_path = "../manual_created_markdwons_for_prompt_design/table_1.md"
    llm_parsed_tables_dir = Path("../llm_parsed_tables/llama_3_1")
    for paper_path in Path("../papers/research_papers").iterdir():
        if paper_path.suffix == '.pdf':
            paper_name_path = Path(f"../llm_parsed_tables/llama_3_1/{paper_path.stem}")
            create_dir_if_not_exists(paper_name_path)
            papers_section_text_path = os.path.join(f"../parsing_experiments/20_10_2024/{paper_path.stem}", "extracted_text_dict.json")
            papers_section_text = read_json(Path(papers_section_text_path))
            for section in papers_section_text:
                markdown_section_file_path = save_str_as_markdown(f"{section}.md", papers_section_text[section])
                parse_markdown_with_mistral(markdown_file_path=f"{section}.md",
                                            output_dir=paper_name_path, model_name=model_name)
                os.remove(f"{section}.md")

        else:
            print(f"Broken file: {paper_path}")
    # for markdown_file in [f"../manual_created_markdwons_for_prompt_design/table_{i}.md" for i in range(1,4)]:
    #     parse_markdown_with_mistral(markdown_file_path=markdown_file,
    #                                 output_dir=llm_parsed_tables_dir, model_name=model_name)