import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tqdm import tqdm

from prompts.parser_prompts import triplets_extraction_prompt_gpt_4
from src.logger import logger
from src.openai_client import get_openai_model_response
from src.parsers.llm_parser import (parse_model_response,
                                    send_request_to_the_model_with_ollama)
from src.parsers.parser import extract_pdf_sections_content
from src.utils import (create_dir_if_not_exists, read_json,
                       read_markdown_file_content,
                       remove_table_data_from_markdown, save_data_to_json_file,
                       save_str_as_markdown, save_str_as_txt_file)

MODEL_NAME = "gpt-4-turbo"
load_dotenv()


def remove_table_content_from_parsed_md_file():
    papers_without_tables_dir = (
        f"parsing_experiments/30_11_2024_without_extracted_table_content_in_section"
    )
    create_dir_if_not_exists(Path(papers_without_tables_dir))

    parsed_papers_dir_name = (
        "parsing_experiments/11_11_2024_fixed_section_extraction_llama3.1"
    )
    papers_dir_list = list(Path(parsed_papers_dir_name).iterdir())

    for paper_path in (pbar := tqdm(papers_dir_list)):
        pbar.set_description(f"Processing document: {paper_path.name}")

        paper_name = paper_path.name
        marker_output_md_file = (
            f"{str(paper_path)}/marker_output/{paper_name}/{paper_name}.md"
        )
        marker_output_metadata_file_path = (
            f"{str(paper_path)}/marker_output/{paper_name}/{paper_name}_meta.json"
        )
        paper_without_table_path_dir = os.path.join(
            papers_without_tables_dir, paper_name
        )
        create_dir_if_not_exists(Path(paper_without_table_path_dir))
        paper_without_table_path = os.path.join(
            paper_without_table_path_dir, paper_name + ".md"
        )
        remove_table_data_from_markdown(marker_output_md_file, paper_without_table_path)

        extracted_sections = extract_pdf_sections_content(
            paper_without_table_path, marker_output_metadata_file_path
        )
        save_data_to_json_file(extracted_sections, paper_without_table_path_dir)


def extract_triplet_from_section(
    markdown_file_path: str | Path,
    output_dir: str | Path,
    prompt_template: str,
    url: str = "http://localhost:11434/api/generate",
    model_name: str = "mistral",
    use_openai: bool = False,
):
    logger.info(f"Extracting triplet with {model_name} ...")
    file_content = read_markdown_file_content(markdown_file_path)
    if file_content:
        output_parser = JsonOutputParser()

        prompt = PromptTemplate(template=prompt_template).format(
            part_of_research_paper=file_content
        )

        if use_openai:
            model_response = get_openai_model_response(prompt, model_name=model_name)
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


if __name__ == "__main__":
    parsed_papers_without_table_content_dir = "parsing_experiments/1_12_2024_llama3.1"
    parsed_papers_without_table_content = list(
        Path(parsed_papers_without_table_content_dir).iterdir()
    )
    triplets_output_dir_without_table_content = (
        f"triplets_extraction/from_each_section_with_table_{MODEL_NAME}"
    )
    create_dir_if_not_exists(Path(triplets_output_dir_without_table_content))

    for paper_path in parsed_papers_without_table_content:
        paper_name_output_path = Path(
            f"{triplets_output_dir_without_table_content}/{paper_path.name}"
        )
        create_dir_if_not_exists(paper_name_output_path)
        papers_section_text_path = os.path.join(
            paper_path,
            "extracted_text_dict.json",
        )
        papers_section_text = read_json(Path(papers_section_text_path))
        for section in papers_section_text:
            save_str_as_markdown(f"{section}.md", papers_section_text[section])
            extract_triplet_from_section(
                markdown_file_path=f"{section}.md",
                output_dir=paper_name_output_path,
                model_name=MODEL_NAME,
                prompt_template=triplets_extraction_prompt_gpt_4,
                use_openai=True,
            )
            os.remove(f"{section}.md")
