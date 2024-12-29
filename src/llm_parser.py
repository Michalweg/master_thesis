import json
import os
import sys
from io import StringIO
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import requests
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from src.logger import logger
from src.openai_client import get_openai_model_response
from src.parser_prompts import (gpt_4_prompt, llama_31_7B_prompt,
                                triplets_extraction_prompt_llama_3_1, gpt_4o_prompt_rana)
from src.utils import (create_dir_if_not_exists, read_json,
                       read_markdown_file_content, save_dict_to_json,
                       save_str_as_markdown, save_str_as_txt_file)

SUPPORTED_MODELS_PROMPTS = {
    "llama3.1": llama_31_7B_prompt,
    "llama3.2": llama_31_7B_prompt,
    "gpt-4o": gpt_4_prompt,
    "gpt-4": gpt_4o_prompt_rana#gpt_4_prompt
}

class ModelResponseTablesExtraction(BaseModel):
    tables_dict: dict = Field(
        description="Python dictionary containing key in form of 'table_1', 'table_2, etc. and values"
        "as strings containing csv file for each extracted table"
    )

def send_request_to_the_model_with_ollama(prompt: str,
                                          model_name: str,
                                          url: str = "http://localhost:11434/api/generate") -> str:

    import subprocess
    data = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0
    })
    # Curl command split into a list
    curl_command = [
        "curl",
        "http://localhost:11434/api/generate",
        "-d", data,
        "-H", "Content-Type: application/json"
    ]
    # Run the curl command
    result = subprocess.run(curl_command, text=True, capture_output=True)
    r = json.loads(result.stdout)['response']

    # r = requests.post(url, json=payload)
    #
    # # Parsing and saving LLM response
    # r = json.loads(r.content)["response"]

    return r



def parse_markdown_with_llm(
    markdown_file_path: str | Path,
    output_dir: str | Path,
    url: str = "http://localhost:11434/api/generate",
    model_name: str = "mistral",
):
    logger.info(f"Parsing Markdown with {model_name} ...")
    file_content = read_markdown_file_content(markdown_file_path)

    if len(file_content) > 10:
        output_parser = JsonOutputParser(pydantic_object=ModelResponseTablesExtraction)
        prompt_template = SUPPORTED_MODELS_PROMPTS[model_name]

        prompt = PromptTemplate(template=prompt_template).format(
            markdown_file_content=file_content
        )

        if model_name == "gpt-4o" or model_name == "gpt-4":
            model_response = get_openai_model_response(prompt, model_name)
        else:
            model_response = send_request_to_the_model_with_ollama(prompt, url)

        if model_response:
                parsed_response_file_path = parse_model_response(model_response, output_parser, output_dir, file_content, markdown_file_path)
                if parsed_response_file_path:
                    dfs = json_csv_to_dataframe_converter(parsed_response_file_path)
                    if dfs:
                        for i, df in enumerate(dfs):
                            df.to_csv(os.path.join(os.path.dirname(parsed_response_file_path), Path(parsed_response_file_path).stem + f"{str(i)}.csv"), index=False)
                    else:
                        base_path, file_name = os.path.split(parsed_response_file_path)
                        save_str_as_markdown(os.path.join(base_path, file_name.split(".")[0]+ "md"), "No table could be extracted from this: \n" + model_response)
        else:
            logger.info(f"The model didn't generate any response")
            save_str_as_txt_file(
                txt_file_path=os.path.join(
                    str(output_dir),
                    Path(markdown_file_path).stem + f"len_{len(file_content)}" + ".txt",
                ),
                str_content=model_response,
            )
        logger.info(f"Parsing Markdown with {model_name} done")
    else:
        logger.warning(f"There is no content in this markdown file: {markdown_file_path}")


def parse_model_response(model_response: str, output_parser: JsonOutputParser | None,
                         output_dir: str | Path, file_content: str, markdown_file_path: str) -> str:
    parsed_response_file_path = ""
    try:
        parsed = output_parser.parse(model_response)
    except Exception as e:
        logger.warning(f"Saving extracted response failed due to an error: {e}")
        save_str_as_txt_file(
            txt_file_path=os.path.join(
                str(output_dir),
                Path(markdown_file_path).stem + f"len_{len(file_content)}" + ".txt",
            ),
            str_content=model_response,
        )
    else:
        if parsed:
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
        else:
            logger.info(f"The parsed response from: {markdown_file_path} is empty!")

    return parsed_response_file_path



def parse_markdown_sections_to_extract_tables_using_llm(
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
    if all_dfs_in_json_file:
        for table_data in all_dfs_in_json_file:
            if isinstance(table_data, dict):
                data = table_data['data']
            elif isinstance(table_data, str):
                data = all_dfs_in_json_file[table_data]
            else:
                data = ""

            if data:
                try:
                    df = pd.read_csv(StringIO(data), sep=",")
                    dfs.append(df)
                except Exception as e:
                    logger.warning(
                        f"Reading dataframe from extracted response: {all_dfs_in_json_file[table_data]} failed due to an error: {e}"
                    )
            else:
                logger.info(f"Extracted section: {str(json_file_path).split('_')[0]} does not contain any table")
    return dfs


if __name__ == "__main__":

    model_name = "llama3.1"
    markdown_file_path = "../manual_created_markdowns_for_prompt_design/table_1.md"
    llm_parsed_tables_dir = Path("../llm_parsed_tables/llama_3_1")
    paper_output_dir_name = "11_11_2024_fixed_section_extraction_llama3.1"


    for paper_path in Path("../papers/research_papers").iterdir():
        if paper_path.suffix == ".pdf":
            paper_name_output_path = Path(f"../parsing_experiments/triplets_extraction/from_each_section_without_table/{paper_path.stem}")
            create_dir_if_not_exists(paper_name_output_path)
            papers_section_text_path = os.path.join(
                f"../parsing_experiments/{paper_output_dir_name}/{paper_path.stem}",
                "extracted_text_dict.json",
            )
            papers_section_text = read_json(Path(papers_section_text_path))

        else:
            print(f"Broken file: {paper_path}")

    #### Running experiment only on manually created files  ####
    # for markdown_file in [f"../manual_created_markdowns_for_prompt_design/table_{i}.md" for i in range(1,4)]:
    #     parse_markdown_with_mistral(markdown_file_path=markdown_file,
    #                                 output_dir=llm_parsed_tables_dir, model_name=model_name)
