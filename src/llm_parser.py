import json
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from src.logger import logger
from src.parser_prompts import gpt_4_prompt, llama_31_7B_prompt
from src.utils import (
    create_dir_if_not_exists,
    read_json,
    read_markdown_file_content,
    save_dict_to_json,
    save_str_as_markdown,
    save_str_as_txt_file,
)

SUPPORTED_MODELS_PROMPTS = {
    "llama3.1": llama_31_7B_prompt,
    "gpt4": gpt_4_prompt
}

class ModelResponse(BaseModel):
    tables_dict: dict = Field(
        description="Python dictionary containing key in form of 'table_1', 'table_2, etc. and values"
        "as strings containing csv file for each extracted table"
    )


def parse_markdown_with_llm(
    markdown_file_path: str | Path,
    output_dir: str | Path,
    url: str = "http://localhost:11434/api/generate",
    model_name: str = "mistral",
):
    logger.info(f"Parsing Markdown with {model_name} ...")
    file_content = read_markdown_file_content(markdown_file_path)

    output_parser = JsonOutputParser(pydantic_object=ModelResponse)
    prompt_template = SUPPORTED_MODELS_PROMPTS[model_name]

    prompt = PromptTemplate(template=prompt_template).format(
        markdown_file_content=file_content
    )
    # partial_variables={"format_instructions": output_parser.get_format_instructions()})

    # prompt = PARSER_PROMPT_TEMPLATE.replace("{markdown_file_content}", file_content)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,
    }
    r = requests.post(url, json=payload)

    # Parsing and saving LLM response
    r = json.loads(r.content)["response"]

    if r:
        try:
            parsed = output_parser.parse(r)
        except Exception as e:
            logger.warning(f"Saving extracted response failed due to an error: {e}")
            save_str_as_txt_file(
                txt_file_path=os.path.join(
                    str(output_dir),
                    Path(markdown_file_path).stem + f"len_{len(file_content)}" + ".txt",
                ),
                str_content=r,
            )
        else:
            parsed_response_file_path = os.path.join(
                str(output_dir),
                Path(markdown_file_path).stem
                + f"len_{len(file_content)}"
                + ".json",
            )
            save_dict_to_json(
                parsed,
                parsed_response_file_path
            )
            dfs = json_csv_to_dataframe_converter(parsed_response_file_path)
            if dfs:
                for i, df in enumerate(dfs):
                    df.to_csv(os.path.join(os.path.dirname(parsed_response_file_path), Path(parsed_response_file_path).stem + f"{str(i)}.csv"), index=False)
            else:
                save_str_as_markdown(Path(parsed_response_file_path).stem + "md", "No table could be extracted from this: \n" + r)
    else:
        logger.info(f"The model didn't generate any response")
        save_str_as_txt_file(
            txt_file_path=os.path.join(
                str(output_dir),
                Path(markdown_file_path).stem + f"len_{len(file_content)}" + ".txt",
            ),
            str_content=r,
        )
    logger.info(f"Parsing Markdown with {model_name} done")


def parse_markdown_sections(
    sections_dict: dict, output_dir: str | Path, model_name: str
):
    for section in sections_dict.keys():
        section_content = sections_dict[section]
        save_str_as_markdown(f"{section}.md", section_content)
        parse_markdown_with_llm(f"{section}.md", output_dir, model_name=model_name)
        os.remove(f"{section}.md")


def json_csv_to_dataframe_converter(json_file_path) -> list[pd.DataFrame]:
    all_dfs_in_json_file = read_json(json_file_path)
    dfs = []
    for table_name in all_dfs_in_json_file:
        if all_dfs_in_json_file[table_name]:
            try:
                df = pd.read_csv(StringIO(all_dfs_in_json_file[table_name]), sep=",")
                dfs.append(df)
            except Exception as e:
                logger.warning(
                    f"Reading dataframe from extracted response: {all_dfs_in_json_file[table_name]} failed due to an error: {e}"
                )
        else:
            logger.info(f"Extracted section: {str(json_file_path).split('_')[0]} does not contain any table")

    return dfs


if __name__ == "__main__":

    model_name = "llama3.1"
    markdown_file_path = "../manual_created_markdowns_for_prompt_design/table_1.md"
    llm_parsed_tables_dir = Path("../llm_parsed_tables/llama_3_1")

    for paper_path in Path("../papers/research_papers").iterdir():
        if paper_path.suffix == ".pdf":
            paper_name_path = Path(f"../llm_parsed_tables/llama_3_1/{paper_path.stem}")
            create_dir_if_not_exists(paper_name_path)
            papers_section_text_path = os.path.join(
                f"../parsing_experiments/20_10_2024/{paper_path.stem}",
                "extracted_text_dict.json",
            )
            papers_section_text = read_json(Path(papers_section_text_path))
            for section in papers_section_text:
                save_str_as_markdown(f"{section}.md", papers_section_text[section])
                parse_markdown_with_llm(
                    markdown_file_path=f"{section}.md",
                    output_dir=paper_name_path,
                    model_name=model_name,
                )
                os.remove(f"{section}.md")

        else:
            print(f"Broken file: {paper_path}")

    #### Running experiment only on manually created files  ####
    # for markdown_file in [f"../manual_created_markdowns_for_prompt_design/table_{i}.md" for i in range(1,4)]:
    #     parse_markdown_with_mistral(markdown_file_path=markdown_file,
    #                                 output_dir=llm_parsed_tables_dir, model_name=model_name)
