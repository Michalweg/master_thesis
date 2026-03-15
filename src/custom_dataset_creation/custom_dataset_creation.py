import io
import os
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from PyPDF2 import PdfReader

from src.const import TASK_NAME
from src.logger import logger
from src.utils import (create_dir_if_not_exists, download_pdf,
                       extract_tables_from_markdown, save_dict_to_json, read_json,
                       normalize_results_in_tdm_list)
from pydantic import BaseModel, Field


def is_pdf_readable(pdf_path: str) -> tuple[bool, str]:
    """
    Check if a PDF file can be opened and read.

    Returns:
        tuple: (is_readable, error_message)
    """
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            # Try to access the number of pages to verify the PDF is valid
            _ = len(reader.pages)
        return True, ""
    except Exception as e:
        return False, str(e)

"""
===================================================================================
INSTRUCTIONS FOR RUNNING THIS CODE:
===================================================================================
Before running this script, you need to make the following changes:

1. CHANGE THE DATASETS_FOR_MARKDOWN VARIABLE:
   - Locate the `datasets_for_markdown` variable in the `create_custom_data` function
   - Update it with the list of datasets you want to process
   - Example: datasets_for_markdown = ["LC-QuAD v2", "QALD-9"]

2. CHANGE THE LINK IN download_github_file FUNCTION CALL:
   - Update the GitHub URL in the call to `download_github_file` to point to
     the correct leaderboard markdown file you want to download
"""


def download_github_file(github_url, output_path):
    """
    Download a file from GitHub and save it to a specified path.

    Args:
        github_url: GitHub blob URL (e.g., https://github.com/KGQA/leaderboard/blob/v2.0/dbpedia/LC-QuAD%20v1.md)
        output_path: Full path where the file should be saved (including filename)
    """
    # Convert GitHub blob URL to raw URL
    if "github.com" in github_url and "/blob/" in github_url:
        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    else:
        raw_url = github_url

    logger.info(f"Downloading from: {raw_url}")

    # Download the file
    response = requests.get(raw_url)
    response.raise_for_status()  # Raise an error for bad status codes

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the file
    with open(output_path, 'wb') as f:
        f.write(response.content)

    logger.info(f"File saved to: {output_path}")
    return output_path


def setup():
    # Set up headless Chrome
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    return driver


def get_rendered_page_source(driver: webdriver, url) -> str:
    # Load the page
    driver.get(url)
    # Wait for the JavaScript to render the table (adjust time as needed or use WebDriverWait)
    time.sleep(3)
    return driver.page_source


def extract_relevant_info_from_rendered_page(page_source: str):
    # Get the rendered HTML
    soup = BeautifulSoup(page_source, "html.parser")
    tabulator_table = soup.find(class_="tabulator-table")

    rows = []
    if tabulator_table:
        for record in tabulator_table.children:
            row_dict = {}
            for row in record:
                if "tabulator-field" in row.attrs:
                    corresponding_text = row.text
                    if corresponding_text and corresponding_text != "-":
                        row_dict[row["tabulator-field"]] = row.text

                if row.find("a"):
                    element = row.find("a")
                    row_dict["PaperUrl"] = element["href"]
                    row_dict["PaperName"] = Path(element["href"]).name

            rows.append(row_dict)

    return rows


def preprocess_data(
    data: Union[list[dict], pd.DataFrame], dataset_name: str, columns_to_drop: list[str], metric_names: list[str]
) -> pd.DataFrame:
    if isinstance(data, list):
        df = pd.DataFrame.from_records(data)
    elif isinstance(data, pd.DataFrame):
        df = data
        df.columns = [col.strip() for col in df.columns]
    else:
        raise TypeError(f"Unsupported type: {type(data)}")

    df["Dataset"] = dataset_name
    columns_to_drop_in_df = set(df.columns).intersection(columns_to_drop)
    df = df.drop(columns=list(columns_to_drop_in_df), axis=1)
    # Ensure that for each metric there's a separate row
    present_metrics = set(metric_names).intersection(df.columns)
    if not present_metrics:
        present_metrics = [col for col in df.columns if col.lower() in [metric.lower() for metric in known_metrics]]

    df_long = df.melt(
        id_vars=[col for col in df.columns if col not in present_metrics],
        value_vars=list(present_metrics),
        var_name="Metric",
        value_name="Result"
    )

    for col in df_long.columns:
        df_long[col] = df_long[col].astype(str).str.strip()
    df_long.replace("-", np.nan, inplace=True)
    return df_long.dropna(ignore_index=True)

def preprocess_table(table: pd.DataFrame) -> pd.DataFrame:
    # Extract url
    reported_column = "Reported By" if "Reported By" in table.columns else "Reported by"
    table["PaperUrl"] = table[reported_column].str.extract(r"\((.*?)\)")

    # Drop rows where no URL could be extracted
    missing = table["PaperUrl"].isna().sum()
    if missing:
        logger.warning(f"Dropping {missing} row(s) with no URL in '{reported_column}'")
    table = table.dropna(subset=["PaperUrl"]).reset_index(drop=True)

    # Extract file name from url (strip trailing slashes to handle URLs like https://example.com/paper-id/)
    table["PaperName"] = table["PaperUrl"].apply(lambda x: os.path.basename(x.rstrip('/')))

    return table


def preprocess_md_file_from_repository(markdown_file_path: str, dataset_name: str,columns_to_drop: list[str],
                                       known_metrics:list[str], papers_dir: str) -> tuple[dict, list[dict]]:
    tables = extract_tables_from_markdown(markdown_file_path)
    if not tables:
        with open(markdown_file_path, "r") as f:
            raw_lines = f.readlines()
        # Strip YAML front matter
        if raw_lines and raw_lines[0].strip() == "---":
            end = next((i for i, l in enumerate(raw_lines[1:], 1) if l.strip() == "---"), None)
            if end is not None:
                raw_lines = raw_lines[end + 1:]
        table_content = "".join(l for l in raw_lines if l.strip().startswith("|"))
        table = pd.read_csv(io.StringIO(table_content), sep="|")
        columns_to_drop = [col for col in table if "unnamed" in col.lower()]
        table.drop(columns=columns_to_drop, inplace=True)
        table.columns = [col.strip() for col in table.columns]
        tables = [table]
    tables_results = {}
    failed_downloads = []

    for table in tables:
        table = preprocess_table(table)
        preprocessed_table = preprocess_data(table, dataset_name=dataset_name, columns_to_drop=columns_to_drop, metric_names=known_metrics)
        if preprocessed_table.empty:
            raise Exception("The table was not extracted, it's empty")
        model_system_column_name = "Model / System" if "Model / System" in preprocessed_table.columns else "Model"
        for paper_url in preprocessed_table["PaperUrl"].unique():
            logger.info(f"Processing paper '{paper_url}'")
            paper_name = Path(paper_url.rstrip('/')).name
            if ".pdf" not in paper_name:
                logger.warning(f"Weird paper name '{paper_name}'")
                paper_name += f".pdf"

            pdf_path = os.path.join(papers_dir, dataset_name, paper_name)
            try:
                download_pdf(paper_url, pdf_path)
                # Verify the PDF can be opened
                is_readable, read_error = is_pdf_readable(pdf_path)
                if not is_readable:
                    logger.error(f"PDF {paper_name} downloaded but cannot be opened: {read_error}")
                    failed_downloads.append({
                        "paper_name": paper_name,
                        "paper_url": paper_url,
                        "error": f"PDF unreadable: {read_error}"
                    })
                    # Delete the unreadable PDF
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                        logger.info(f"Deleted unreadable PDF: {pdf_path}")
            except Exception as e:
                error_message = str(e)
                logger.error(f"Failed to download {paper_name} from {paper_url}: {error_message}")
                failed_downloads.append({
                    "paper_name": paper_name,
                    "paper_url": paper_url,
                    "error": error_message
                })

        result = (
            preprocessed_table.groupby("PaperName")
            .apply(lambda g: g[["Dataset", model_system_column_name, "Metric", "Result", "PaperUrl"]]
                   .rename(columns={
                model_system_column_name: "Model",
                "Metric": "Metric",
                "Result": "Result"
            })
                   .to_dict(orient="records"))
            .to_dict()
        )
        tables_results.update(result)

    # Exclude papers that failed to download from the result
    # Normalize names by removing .pdf extension for comparison (handles inconsistent naming)
    failed_paper_names = {download["paper_name"].replace(".pdf", "") for download in failed_downloads}
    tables_results = {k: v for k, v in tables_results.items() if k.replace(".pdf", "") not in failed_paper_names}

    return tables_results, failed_downloads


def create_result_dict_in_correct_format(result: dict) -> dict:
    correct_result = {}
    for paper_name in result.keys():
        if "TDMs" in result[paper_name]:
            return result
        if not paper_name.endswith(".pdf"):
            correct_result[paper_name + ".pdf"] = {"PaperURL": result[paper_name][0].get("PaperUrl", ""),
                                          "TDMs": result[paper_name]}
        else:
            correct_result[paper_name] = {"PaperURL": result[paper_name][0].get("PaperUrl", ""),
                                         "TDMs": result[paper_name]}
    return correct_result

def add_hardcoded_task_to_result_dict(result: dict, hardcoded_task_name: str = TASK_NAME) -> dict: # MOVE IT TO CONST.py
    for paper_name in result.keys():
        for i, papers_tdm in enumerate(result[paper_name]["TDMs"]):
            result[paper_name]["TDMs"][i]["Task"] = hardcoded_task_name
    return result


def normalize_results_in_result_dict(result: dict) -> dict:
    """
    Normalize all Result values in the result dict to 0-1 decimal scale.

    Args:
        result: Dictionary with structure {paper_name: {"TDMs": [...]}}

    Returns:
        Dictionary with normalized Result values
    """
    for paper_name in result.keys():
        if "TDMs" in result[paper_name]:
            result[paper_name]["TDMs"] = normalize_results_in_tdm_list(result[paper_name]["TDMs"])
    return result


def mark_authors_approach(result: dict, model_name: str, paper_name: str) -> dict:
    """
    Add "authors_approach": "yes" to all TDMs matching a given model within a specific paper.

    Args:
        result: Dictionary with structure {paper_name: {"TDMs": [...]}}
        model_name: The model name to match (e.g. "WDAqua-core1")
        paper_name: The paper key to target (e.g. "swj2038.pdf")

    Returns:
        Updated result dictionary
    """
    if paper_name not in result:
        logger.warning(f"Paper '{paper_name}' not found in result dict.")
        return result
    for tdm in result[paper_name].get("TDMs", []):
        if tdm.get("Model") == model_name and "authors_approach" not in tdm:
            tdm["authors_approach"] = "yes"
    return result


def combine_refined_authors_approach_files(base_dir: str, output_path: str) -> dict:
    """
    Combine all files containing 'refined_authors_approach' in their name across all
    subdirectories of base_dir, keeping only TDMs that have "authors_approach": "yes".

    Args:
        base_dir: Root directory to search (e.g. "custom_dataset_papers_refined/dbpedia")
        output_path: Path where the combined JSON will be saved

    Returns:
        Combined dictionary with structure {paper_name: {"PaperURL": ..., "TDMs": [...]}}
    """
    combined: dict = {}
    for fpath in sorted(Path(base_dir).rglob("*.json")):
        if "refined_authors_approach" not in fpath.name:
            continue
        data = read_json(str(fpath))
        for paper, content in data.items():
            filtered = [tdm for tdm in content.get("TDMs", []) if tdm.get("authors_approach") == "yes"]
            if filtered:
                if paper not in combined:
                    combined[paper] = {"PaperURL": content.get("PaperURL", ""), "TDMs": []}
                combined[paper]["TDMs"].extend(filtered)
        logger.info(f"Processed {fpath.name}")

    save_dict_to_json(combined, output_path)
    logger.info(f"Combined authors-approach file saved to '{output_path}' ({len(combined)} papers).")
    return combined


def combine_all_authors_approach_files(base_dir: str, output_path: str, exclude_dirs: list[str] = None) -> dict:
    """
    Iterate over all immediate subdirectories of base_dir (excluding specified dirs),
    find files matching '*refined_authors_approach*.json' in each, and combine all TDMs
    into a single output file.

    Args:
        base_dir: Root directory to search (e.g. "custom_dataset_papers_refined/dbpedia")
        output_path: Path where the combined JSON will be saved
        exclude_dirs: List of subdirectory names to skip (default: ["all_papers_dbpedia"])

    Returns:
        Combined dictionary with structure {paper_name: {"PaperURL": ..., "TDMs": [...]}}
    """
    if exclude_dirs is None:
        exclude_dirs = ["all_papers_dbpedia"]

    combined: dict = {}
    for subdir in sorted(Path(base_dir).iterdir()):
        if not subdir.is_dir() or subdir.name in exclude_dirs:
            continue
        matches = list(subdir.glob("*refined_authors_approach*.json"))
        if not matches:
            logger.info(f"Skipping '{subdir.name}' (no matching file)")
            continue
        data = read_json(str(matches[0]))
        for paper, content in data.items():
            if paper not in combined:
                combined[paper] = {"PaperURL": content.get("PaperURL", ""), "TDMs": []}
            combined[paper]["TDMs"].extend(normalize_results_in_tdm_list(content.get("TDMs", [])))
        logger.info(f"Merged {matches[0].name} ({len(data)} papers)")

    save_dict_to_json(combined, output_path)
    logger.info(f"Saved combined file to '{output_path}' ({len(combined)} papers).")
    return combined


if __name__ == "__main__":
    columns_to_drop = ["Year", "Language", "Reported by", "id"]
    known_metrics = ["F1", "Precision", "Recall", "Hits@1", "Hits@10", "Precision@1", "MRR", "Hits@5", "Accuracy"]
    custom_dataset_papers_dir = "custom_dataset_papers_refined"
    create_dir_if_not_exists(Path(custom_dataset_papers_dir))
    analyzed_knowledge_graph = "wikidata"

    datasets_for_markdown = ["TimeQuestions - Temporal Answer"] # custom_dataset_papers/dbpedia/LC-QuAD v1/LC-QuAD v1.md
    for dataset in datasets_for_markdown:
        markdown_file = os.path.join(custom_dataset_papers_dir, analyzed_knowledge_graph, dataset, dataset + ".md")
        download_github_file(f"https://github.com/KGQA/leaderboard/blob/v2.0/wikidata/{dataset}.md", markdown_file)
        preprocessed_dict, failed_downloads = preprocess_md_file_from_repository(markdown_file, dataset_name=dataset, columns_to_drop=columns_to_drop, known_metrics=known_metrics, papers_dir=os.path.join(custom_dataset_papers_dir, analyzed_knowledge_graph))
        result = create_result_dict_in_correct_format(preprocessed_dict)
        result = add_hardcoded_task_to_result_dict(result)
        result = normalize_results_in_result_dict(result)
        save_dict_to_json(result, Path(markdown_file).with_suffix(".json"))

        # Save failed downloads if there are any
        if failed_downloads:
            failed_downloads_file = os.path.join(os.path.dirname(markdown_file), "failed_downloads.json")
            save_dict_to_json(failed_downloads, failed_downloads_file)
            logger.info(f"Saved {len(failed_downloads)} failed downloads to {failed_downloads_file}")

    # Collect all unique PDFs from dbpedia subdirectories into a single flat directory
    all_papers_dir = Path(custom_dataset_papers_dir) / analyzed_knowledge_graph / f"all_papers_{analyzed_knowledge_graph}"
    create_dir_if_not_exists(all_papers_dir)
    dbpedia_dir = Path(custom_dataset_papers_dir) / analyzed_knowledge_graph
    copied, skipped = 0, 0
    for pdf_path in dbpedia_dir.rglob("*.pdf"):
        dest = all_papers_dir / pdf_path.name
        if not dest.exists():
            shutil.copy2(pdf_path, dest)
            copied += 1
        else:
            skipped += 1
    logger.info(f"Collected PDFs into '{all_papers_dir}': {copied} copied, {skipped} skipped (duplicates).")

    # Combine all per-dataset JSON files into one merged JSON in all_papers_dbpedia
    combined: dict = {}
    for subdir in sorted(dbpedia_dir.iterdir()):
        if not subdir.is_dir():
            continue
        dataset_json = subdir / f"{subdir.name}.json"
        if not dataset_json.exists():
            logger.warning(f"No dataset JSON found at {dataset_json}, skipping.")
            continue
        data = read_json(str(dataset_json))
        for paper_key, paper_value in data.items():
            if paper_key not in combined:
                combined[paper_key] = {"PaperURL": paper_value["PaperURL"], "TDMs": []}
            combined[paper_key]["TDMs"].extend(paper_value["TDMs"])
        logger.info(f"Merged {dataset_json.name} ({len(data)} papers)")

    combined = normalize_results_in_result_dict(combined)
    combined_output_path = all_papers_dir / "all_dbpedia.json"
    save_dict_to_json(combined, str(combined_output_path))
    logger.info(f"Combined JSON saved to '{combined_output_path}' ({len(combined)} unique papers).")
