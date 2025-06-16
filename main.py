import logging
import os
from pathlib import Path

from tqdm import tqdm

from src.llm_parser import parse_markdown_sections_to_extract_tables_using_llm
from src.logger import logger
from src.marker_parser import parse_pdf_with_marker
from src.parser import (extract_pdf_sections_content,
                        parse_pdf_with_llama_parse, parse_pdf_with_pdf_plumber,
                        parse_pdf_with_tabula)
from src.utils import (create_dir_if_not_exists, extract_tables_from_markdown,
                       save_data_to_json_file, saving_list_of_dfs, set_env)

# Suppress warnings from font handling
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

set_env()
MODEL_NAME = "gpt-4o"

if __name__ == "__main__":

    # papers_dir = "papers/research_papers"
    papers_dir = 'various_tables_extraction_approaches_paper_dir'
    experiment_dir = f"various_tables_extraction_approaches_summary_dir"
    create_dir_if_not_exists(Path(experiment_dir))
    papers_dir_list = list(Path(papers_dir).iterdir())

    for paper_path in (pbar := tqdm(papers_dir_list)):
        pbar.set_description(f"Processing document: {paper_path.name}")
        if paper_path.suffix == ".pdf":
            # paper_path = "papers/research_papers/2406.04383v2.pdf"
            paper_name = paper_path.stem
            marker_output_dir = f"{experiment_dir}/{paper_name}/marker_output"

            # Extracts markdown file out of PDF
            markdown_file_path = parse_pdf_with_marker(
                str(paper_path), marker_output_dir
            )
            # Extract sections of PDF
            pdf_sections_content = extract_pdf_sections_content(markdown_file_path)
            # Extract tables with Tabula
            extracted_tables_tabula = parse_pdf_with_tabula(paper_path)
            # Parsing PDF with Lama Parse
            # file_content = parse_pdf_with_llama_parse(paper_path, f"{experiment_dir}/{paper_name}/lama_markdown.md")

            # Parsing PDF with PDF plumber
            pdf_plumber_tables = parse_pdf_with_pdf_plumber(paper_path)


            paper_output_dir_path = Path(experiment_dir) / Path(paper_path).stem
            os.makedirs(paper_output_dir_path, exist_ok=True)
            save_data_to_json_file(pdf_sections_content, paper_output_dir_path)

            try:
                extracted_tables_with_markdown = extract_tables_from_markdown(
                    markdown_file_path
                )
                # Saving manual extracted tables
                manual_tables_path = os.path.join(
                    paper_output_dir_path, "manual_extracted_tables"
                )
                os.makedirs(manual_tables_path, exist_ok=True)
                saving_list_of_dfs(
                    extracted_tables_with_markdown, Path(manual_tables_path)
                )

            except Exception as e:
                logger.error(f"Extracting with markdown yielded this error:{e}")

            # Saving tables from tabula
            tabula_tables_file_path = os.path.join(
                paper_output_dir_path, "tabula_extracted_files"
            )
            os.makedirs(tabula_tables_file_path, exist_ok=True)
            saving_list_of_dfs(extracted_tables_tabula, Path(tabula_tables_file_path))

            # Saving tables from pdf plumber
            pdf_plumber_tables_file_path = os.path.join(
                paper_output_dir_path, "pdf_plumber_tables"
            )
            os.makedirs(pdf_plumber_tables_file_path, exist_ok=True)
            try:
                saving_list_of_dfs(pdf_plumber_tables, Path(pdf_plumber_tables_file_path))
            except Exception as e:
                logger.error(f"PDF plumber extracted tables could not been saved, due to: {e}")

            # LLM extracted tables
            llm_tables_file_path = os.path.join(
                paper_output_dir_path, "llm_extracted_tables"
            )
            os.makedirs(llm_tables_file_path, exist_ok=True)
            parse_markdown_sections_to_extract_tables_using_llm(
                pdf_sections_content, llm_tables_file_path, model_name=MODEL_NAME
            )
        else:
            logger.warning(
                f"The provided path does not have required extension: {paper_path.suffix} instead of PDF"
            )
            continue
