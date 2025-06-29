import os
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.logger import logger
from src.openai_client import get_openai_model_response
from src.parsers.llm_parser import (parse_model_response,
                                    send_request_to_the_model_with_ollama)
from src.parsers.marker_parser import parse_pdf_with_marker
from src.parsers.parser import extract_pdf_sections_content
from src.utils import (create_dir_if_not_exists, read_json,
                       read_markdown_file_content, save_dict_to_json,
                       save_str_as_markdown, save_str_as_txt_file)

MODEL_NAME = "gpt-4o"


class AuthorsModelResponse(BaseModel):
    extracted_model_approach_names: list[str] = Field(
        description="List of extracted model/approaches developed by authors of the paper"
    )


EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_PROMPT = """
You will be provided with a section of the paper. Your task is to extract the name of the approach/model developed by the 
authors of the paper. Ideally you should extract the name of this approach/model as it appears in the results table. 
Output only the approaches/models developed by the authors. If there is no information about the approaches/models developed by 
authors in the section, you should output an empty list. 

Here is the section:
{section}

{format_instructions}
"""


def extract_author_model_prediction(
    markdown_file_path: str | Path,
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

        prompt = PromptTemplate(
            template=prompt_template,
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        ).format(section=file_content)

        if use_openai:
            model_response = get_openai_model_response(prompt, model_name=model_name)
            print(model_response)
        else:
            model_response = send_request_to_the_model_with_ollama(
                prompt, model_name, url
            )

        if model_response:
            _ = parse_model_response(
                model_response,
                output_parser,
                output_dir,
                file_content,
                markdown_file_path,
            )
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
        logger.warning(
            f"There is no content in this markdown file: {markdown_file_path}"
        )


def combine_all_sections_based_json_into_one_file(
    output_dir_path: str, result_file_name: str = "author_model_approaches.json"
) -> None:
    output_sections_dict_list: list[dict] = []
    for section_file in Path(output_dir_path).iterdir():
        if section_file.suffix == ".json":
            section_result = read_json(section_file)
            if section_result["extracted_model_approach_names"]:
                output_sections_dict_list.append(
                    {
                        section_file.name.split("len")[0]: section_result[
                            "extracted_model_approach_names"
                        ]
                    }
                )
    save_dict_to_json(
        output_sections_dict_list, os.path.join(output_dir_path, result_file_name)
    )


if __name__ == "__main__":
    author_model_approach_experiment_dir_path = (
        "extending_results_extracton_with_author_approach"
    )
    papers_dir = os.path.join(author_model_approach_experiment_dir_path, "papers")
    parsed_papers_without_table_content: list = list(Path(papers_dir).iterdir())

    # Setting up and creating output dir
    author_model_extraction_dir_without_table_content = f"author_model_extraction/from_each_section_{MODEL_NAME}_no_vector_db_29_04_2025"
    create_dir_if_not_exists(Path(author_model_extraction_dir_without_table_content))

    for paper_path in tqdm(parsed_papers_without_table_content):
        # Setting up and creating an output di for specific paper
        paper_name_output_path = Path(
            f"{author_model_extraction_dir_without_table_content}/{paper_path.stem}"
        )
        create_dir_if_not_exists(paper_name_output_path)

        # Reading extracted_text_dict.json if exists, otherwise create it on the spot
        papers_section_text_path = os.path.join(
            paper_path,
            "extracted_text_dict.json",
        )

        if not Path(papers_section_text_path).exists():
            # Create this file by parsing the file with Marker and then extract section out of it.
            marker_output_dir = f"{author_model_approach_experiment_dir_path}/{paper_path.stem}/marker_output"
            markdown_file_path = parse_pdf_with_marker(
                str(paper_path), marker_output_dir
            )
            papers_section_text = extract_pdf_sections_content(markdown_file_path)
        else:
            papers_section_text = read_json(Path(papers_section_text_path))

        # Iterating through each section
        for section in papers_section_text:
            if papers_section_text[section]:
                save_str_as_markdown(f"{section}.md", papers_section_text[section])
                extract_author_model_prediction(
                    markdown_file_path=f"{section}.md",
                    output_dir=paper_name_output_path,
                    model_name=MODEL_NAME,
                    prompt_template=EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_PROMPT,
                    use_openai=True,
                )
                os.remove(f"{section}.md")
            else:
                logger.warning(
                    f"For this section: {section} no content could be extracted thus no model/approach"
                )

            # Saving all extracted results for given paper
            combine_all_sections_based_json_into_one_file(output_dir_path=paper_path)
