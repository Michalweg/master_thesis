import json
import re
from pathlib import Path

import pandas as pd
import pdfplumber
import tabula
from llama_parse import LlamaParse
from PyPDF2 import PdfReader
from tqdm import tqdm

from src.logger import logger
from src.utils import (
    create_dir_if_not_exists,
    find_closest_string,
    read_markdown_file_content,
    run_bash_command,
    save_str_as_markdown,
)


def parse_pdf_with_tabula(pdf_file_path: str | Path) -> list[pd.DataFrame]:
    logger.info(f"Extracting tables from PDF: {pdf_file_path} with Tabula")
    dfs = []
    reader = PdfReader(pdf_file_path)
    no_pages = len(reader.pages)
    for page in range(no_pages):
        try:
            # Try using lattice mode
            # print(f"Attempting to extract tables with lattice=True ")
            tables = tabula.read_pdf(pdf_file_path, pages=page, lattice=True)
            if tables:
                # print("Tables extracted successfully using lattice=True.")
                dfs += tables
        except Exception as e:
            # print(f"Error with lattice=True on page {page}: {e}")
            # print("Attempting to extract tables with stream=True...")

            # Try using stream mode if lattice fails
            try:
                tables = tabula.read_pdf(pdf_file_path, pages=page, stream=True)
                # print("Tables extracted successfully using stream=True.")
                dfs += tables
            except Exception as e:
                # print(f"Error with stream=True on page {page}: {e}")
                continue
    logger.info(f"Extracting tables from PDF: {pdf_file_path} completed")
    return dfs


def parse_pdf_with_pdf_plumber(pdf_file_path: str | Path):
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            print(page.extract_tables())
            text = page.extract_text()
            print(text)


def parse_pdf_with_py_pdf2(pdf_file_path: str | Path) -> dict[int, str]:
    reader = PdfReader(pdf_file_path)
    documents_text: dict = {}

    for i in range(len(reader.pages)):
        page = reader.pages[i]
        documents_text.update({str(i): page.extract_text()})
    return documents_text


def extract_pdf_section(section_name: str, markdown_file_path: str) -> str:
    section_content = run_bash_command(
        f'markdown-extract "{section_name}" {markdown_file_path}'
    )
    return section_content


def fix_sections_with_wrong_chars(
    marker_markdown_file_path, extracted_sections, wrong_chars=set("*")
) -> list[str]:
    file_content = read_markdown_file_content(marker_markdown_file_path)
    no_of_changes = 0
    for i, section in enumerate(extracted_sections):
        if set(section).intersection(wrong_chars):
            no_of_changes += 1
            correct_section = "".join(
                [char for char in section if char not in wrong_chars]
            )
            file_content = file_content.replace(section, correct_section)
            extracted_sections[i] = correct_section
        elif section.startswith("# "):
            correct_section = section[2:]
            extracted_sections[i] = correct_section
        elif section.startswith("#"):
            extracted_sections[i] = section[1:]
    if no_of_changes > 0:
        save_str_as_markdown(marker_markdown_file_path, file_content)
    return extracted_sections


def extract_pdf_sections_with_markdown(marker_markdown_file_path: str) -> list[str]:
    files_content = read_markdown_file_content(marker_markdown_file_path)
    sections = re.findall("(?<=\##)(.*?)(?=\n)", files_content)
    sections = [
        section if not section.startswith(" ") else section[1:] for section in sections
    ]
    return sections


def extract_pdf_sections_with_marker_metadata(marker_metadata_file: str) -> list[str]:
    with open(marker_metadata_file) as metadata_file:
        files_content = json.load(metadata_file)
    if "toc" in files_content:
        toc_component = files_content["toc"]
    else:
        toc_component = files_content["pdf_toc"]
    sections = [section["title"] for section in toc_component]
    return sections


def combine_section_names(
    section_names_with_correct_names, section_names_with_correct_structure
):
    for i in range(len(section_names_with_correct_structure)):
        # Choose the most similar section name
        closest_correct_section_name = find_closest_string(
            section_names_with_correct_structure[i], section_names_with_correct_names
        )
        section_names_with_correct_structure[i] = closest_correct_section_name
    return section_names_with_correct_structure


def extract_pdf_sections_content(marker_markdown_file_path: str) -> dict[str, str]:
    logger.info("Extracting PDF sections ...")
    pdf_sections_correct_names = extract_pdf_sections_with_markdown(
        marker_markdown_file_path=marker_markdown_file_path
    )
    marker_metadata_file_path = Path(marker_markdown_file_path).parent / (
        Path(marker_markdown_file_path).stem + "_meta.json"
    )
    pdf_sections = extract_pdf_sections_with_marker_metadata(
        str(marker_metadata_file_path)
    )
    combine_sections_names = combine_section_names(
        pdf_sections_correct_names, pdf_sections
    )
    combine_sections_names = fix_sections_with_wrong_chars(
        marker_markdown_file_path, combine_sections_names
    )
    pdf_section_contents = {}
    for section in combine_sections_names:
        pdf_section_contents.update(
            {
                section: extract_pdf_section(
                    section, markdown_file_path=marker_markdown_file_path
                )
            }
        )
        if not section:
            logger.warning(f"The provided section: {section} from the file: {marker_markdown_file_path} is empty!")

    logger.info("PDF sections extracted")
    return pdf_section_contents


def parse_pdf_with_llama_parse(
    pdf_file_path: str | Path, markdown_file_path: str | Path
):
    logger.info("Parsing PDF ... wih Lama Parse")
    documents = LlamaParse(result_type="markdown").load_data(pdf_file_path)
    all_file_content = "\n".join([doc.text for doc in documents])
    save_str_as_markdown(markdown_file_path, all_file_content)
    logger.info("Parsing PDF ... with Lama Parse done")
    return all_file_content


if __name__ == "__main__":
    papers_dir = "papers/research_papers"
    experiment_dir = "parsing_experiments/29_10_2024_tabula_lattice_mode"
    papers_dir_list = list(Path(papers_dir).iterdir())
    create_dir_if_not_exists(Path(experiment_dir))

    for paper_path in (pbar := tqdm(papers_dir_list)):
        pbar.set_description(f"Processing document: {paper_path.name}")
        if paper_path.suffix == ".pdf":
            dfs = parse_pdf_with_tabula(paper_path)
            output_dir_path = Path(experiment_dir) / paper_path.name
            create_dir_if_not_exists(output_dir_path)
            for i, df in enumerate(dfs):
                df.to_csv(
                    output_dir_path / f"{paper_path.name}_{str(i)}.csv", index=False
                )
