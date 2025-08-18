import os
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from src.logger import logger
from src.utils import (create_dir_if_not_exists, download_pdf,
                       extract_tables_from_markdown, save_dict_to_json)


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
        var_name="metric",
        value_name="value"
    )
    df_long.replace("-", np.nan, inplace=True)
    return df_long.dropna(ignore_index=True)

def preprocess_table(table: pd.DataFrame) -> pd.DataFrame:
    # Extract url
    reported_column = "Reported By" if "Reported By" in table.columns else "Reported by"
    table["PaperUrl"] = table[reported_column].str.extract(r"\((.*?)\)")

    # Extract file name from url
    table["PaperName"] = table["PaperUrl"].apply(lambda x: os.path.basename(x))

    return table


def preprocess_md_file_from_repository(markdown_file_path: str, dataset_name: str,columns_to_drop: list[str],
                                       known_metrics:list[str]) -> dict:
    tables = extract_tables_from_markdown(markdown_file_path)
    tables_results = {}
    for table in tables:
        table = preprocess_table(table)
        preprocessed_table = preprocess_data(table, dataset_name=dataset_name, columns_to_drop=columns_to_drop, metric_names=known_metrics)
        model_system_column_nae = "Model / System" if "Model / System" in preprocessed_table.columns else "Model"
        result = (
            preprocessed_table.groupby("PaperName")
            .apply(lambda g: g[["Dataset", model_system_column_nae, "metric", "value"]]
                   .rename(columns={
                model_system_column_nae: "Model",
                "metric": "Metric",
                "value": "Result"
            })
                   .to_dict(orient="records"))
            .to_dict()
        )
        tables_results.update(result)
    return tables_results

if __name__ == "__main__":
    columns_to_drop = ["Year", "Language", "Reported by", "id"]
    known_metrics = ["F1", "Precision", "Recall", "Hits@1", "Hits@10", "Precision@1", "MRR", "Hits@5"]
    custom_dataset_papers_dir = "custom_dataset_papers"


    datasets_for_markdown = ["TimeQuestions - Oridinal", "TimeQuestions - Temporal Answer", "Mintaka"]
    for dataset in datasets_for_markdown:
        markdown_file = os.path.join(custom_dataset_papers_dir, dataset, dataset + ".md")
        preprocessed_dict = preprocess_md_file_from_repository(markdown_file, dataset_name=dataset, columns_to_drop=columns_to_drop, known_metrics=known_metrics)
        save_dict_to_json(preprocessed_dict, Path(markdown_file).with_suffix(".json"))


    manual_work_needed = ["Compositional Wikidata Questions"]


    base_url = "https://kgqa.github.io/leaderboard/datasets/wikidata/"
    datasets = ["MKQA", "RuBQ-v2", "CronQuestions", "Mintaka", "SimpleQuestionsWikidata", "TimeQuestions - Explicit",
                "TimeQuestions - Implicit", "TimeQuestions - Oridinal", "TimeQuestions - Overall",
                "TimeQuestions - Temporal Answer", "QALD-9-Plus-Wikidata"]

    driver = setup()

    already_processed_datasets = [dir_name for dir_name in os.listdir(custom_dataset_papers_dir)]

    # Iterate trough datasets:
    for dataset in datasets:
        if dataset in already_processed_datasets:
            logger.info(f"Skipping dataset '{dataset}' as it's already done")
            continue

        url = base_url + dataset
        page_source = get_rendered_page_source(driver, url)
        data = extract_relevant_info_from_rendered_page(page_source)

        if data:
            preprocessed_data = preprocess_data(data, dataset, columns_to_drop, metric_names=known_metrics)

            dataset_papers_dir = os.path.join(custom_dataset_papers_dir, dataset)
            create_dir_if_not_exists(Path(dataset_papers_dir))

            for paper_url in preprocessed_data["PaperUrl"].unique():
                logger.info(f"Processing paper '{paper_url}'")
                paper_name = Path(paper_url).name
                if ".pdf" not in paper_name:
                    logger.warning(f"Weird paper name '{paper_name}'")
                    paper_name += f".pdf"
                download_pdf(paper_url, os.path.join(dataset_papers_dir, paper_name))

            result = (
                preprocessed_data.groupby("PaperName")
                .apply(lambda g: g[["Dataset", "Model / System", "metric", "value"]]
                       .rename(columns={
                    "Model / System": "Model",
                    "metric": "Metric",
                    "value": "Result"
                })
                       .to_dict(orient="records"))
                .to_dict()
            )
            save_dict_to_json(result, os.path.join(dataset_papers_dir, "result.json"))

        else:
            logger.warning(f"Skipping dataset '{dataset}' due to lack of data")

    driver.quit()
