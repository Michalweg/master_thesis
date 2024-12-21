import os
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from src.llm_parser import (parse_model_response,
                            send_request_to_the_model_with_ollama)
from src.logger import logger
from src.openai_client import get_openai_model_response
from src.parser import extract_pdf_sections_content
from src.utils import (create_dir_if_not_exists, read_json,
                       read_markdown_file_content,
                       remove_table_data_from_markdown, save_data_to_json_file,
                       save_str_as_markdown, save_str_as_txt_file)
from tqdm import tqdm

MODEL_NAME = "gpt-4o"

class AuthorsModelResponse(BaseModel):
    extracted_model_approach_names: list[str] = Field(description="List of extracted model/approaches developed by authors of the paper")

EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_PROMPT = """
You will be provided with a section of the paper. Your task is to extract the name of the approach/model developed by the 
authors of the paper. Ideally you should extract the name of this approach/model as it appears in the results table. 
Output only the approaches/models developed by the authors. If there is no information about the approaches/models developed by 
authors in the section, you should output an empty list. 

Here is the section:
{section}

{format_instructions}
"""


def extract_author_model_prediction( markdown_file_path: str | Path,
    output_dir: str | Path,
    prompt_template: str,
    url: str = "http://localhost:11434/api/generate",
    model_name: str = "mistral",
    use_openai: bool = False,
):
    logger.info(f"Extracting author model {model_name} ...")
    file_content = read_markdown_file_content(markdown_file_path)
    if file_content:
        output_parser = JsonOutputParser(pydantic_object=AuthorsModelResponse)

        prompt = PromptTemplate(template=prompt_template, partial_variables={"format_instructions": output_parser.get_format_instructions()}).format(
            section=file_content
        )

        if use_openai:
            model_response = get_openai_model_response(prompt, model_name=model_name)
            print(model_response)
        else:
            model_response = send_request_to_the_model_with_ollama(prompt, model_name, url)

        if model_response:
            _ = parse_model_response(model_response, output_parser, output_dir, file_content, markdown_file_path)
        else:
            logger.info(f"The model didn't generate any response")
            save_str_as_txt_file(
                txt_file_path=os.path.join(
                    str(output_dir),
                    Path(markdown_file_path).stem + f"len_{len(file_content)}" + ".txt",
                ),
                str_content=model_response,
            )
        logger.info(f"Extracting triplet with {model_name} done")
    else:
        logger.warning(f"There is no content in this markdown file: {markdown_file_path}")


if __name__ == "__main__":
    parsed_papers_without_table_content_dir = "parsing_experiments/15_12_2024_gpt-4o"
    parsed_papers_without_table_content = list(Path(parsed_papers_without_table_content_dir).iterdir())
    author_model_extraction_dir_without_table_content = f"author_model_extraction/from_each_section_{MODEL_NAME}_no_vector_db"
    create_dir_if_not_exists(Path(author_model_extraction_dir_without_table_content))

    for paper_path in parsed_papers_without_table_content:
        paper_name_output_path = Path(f"{author_model_extraction_dir_without_table_content}/{paper_path.name}")
        create_dir_if_not_exists(paper_name_output_path)
        papers_section_text_path = os.path.join(
            paper_path,
            "extracted_text_dict.json",
        )
        papers_section_text = read_json(Path(papers_section_text_path))
        for section in papers_section_text:
            save_str_as_markdown(f"{section}.md", papers_section_text[section])
            extract_author_model_prediction(
                markdown_file_path=f"{section}.md",
                output_dir=paper_name_output_path,
                model_name=MODEL_NAME,
                prompt_template=EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_PROMPT,
                use_openai=True,
            )
            os.remove(f"{section}.md")