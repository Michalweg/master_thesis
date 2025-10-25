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
                       extract_tables_from_markdown, save_dict_to_json, read_json)
from pydantic import BaseModel, Field


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

    # Extract file name from url
    table["PaperName"] = table["PaperUrl"].apply(lambda x: os.path.basename(x))

    return table


def preprocess_md_file_from_repository(markdown_file_path: str, dataset_name: str,columns_to_drop: list[str],
                                       known_metrics:list[str], papers_dir: str) -> dict:
    tables = extract_tables_from_markdown(markdown_file_path)
    if not tables:
        table = pd.read_csv(markdown_file_path, sep="|")
        columns_to_drop = [col for col in table if "unnamed" in col.lower()]
        table.drop(columns=columns_to_drop, inplace=True)
        table.columns = [col.strip() for col in table.columns]
        tables = [table]
    tables_results = {}
    for table in tables:
        table = preprocess_table(table)
        preprocessed_table = preprocess_data(table, dataset_name=dataset_name, columns_to_drop=columns_to_drop, metric_names=known_metrics)
        model_system_column_name = "Model / System" if "Model / System" in preprocessed_table.columns else "Model"
        for paper_url in preprocessed_table["PaperUrl"].unique():
            logger.info(f"Processing paper '{paper_url}'")
            paper_name = Path(paper_url).name
            if ".pdf" not in paper_name:
                logger.warning(f"Weird paper name '{paper_name}'")
                paper_name += f".pdf"
            download_pdf(paper_url, os.path.join(papers_dir, dataset_name, paper_name))
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
    return tables_results


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

def add_hardcoded_task_to_result_dict(result: dict, hardcoded_task_name: str = "question answering") -> dict:
    for paper_name in result.keys():
        for i, papers_tdm in enumerate(result[paper_name]["TDMs"]):
            result[paper_name]["TDMs"][i]["Task"] = hardcoded_task_name
    return result


if __name__ == "__main__":
    columns_to_drop = ["Year", "Language", "Reported by", "id"]
    known_metrics = ["F1", "Precision", "Recall", "Hits@1", "Hits@10", "Precision@1", "MRR", "Hits@5", "Accuracy"]
    custom_dataset_papers_dir = "custom_dataset_papers"
    manual_work_needed = ["Compositional Wikidata Questions"]
    analyzed_knowledge_graph = "dbpedia"

    # for paper in Path(custom_dataset_papers_dir).iterdir():
    #     if paper.is_dir():
    #         result_file_path = paper.joinpath("result.json")
    #         result_dict = read_json(result_file_path)
    #         correct_result_dict = create_result_dict_in_correct_format(result_dict)
    #         correct_result_dict = add_hardcoded_task_to_result_dict(correct_result_dict)
    #         save_dict_to_json(correct_result_dict, result_file_path)

    ### MARKDOWN APPROACH ###
    datasets_for_markdown = ["QALD-9-Plus-DBpedia"] # custom_dataset_papers/dbpedia/LC-QuAD v1/LC-QuAD v1.md
    for dataset in datasets_for_markdown:
        markdown_file = os.path.join(custom_dataset_papers_dir, analyzed_knowledge_graph, dataset, dataset + ".md")
        preprocessed_dict = preprocess_md_file_from_repository(markdown_file, dataset_name=dataset, columns_to_drop=columns_to_drop, known_metrics=known_metrics, papers_dir=os.path.join(custom_dataset_papers_dir, analyzed_knowledge_graph))
        result = create_result_dict_in_correct_format(preprocessed_dict)
        result = add_hardcoded_task_to_result_dict(result)
        save_dict_to_json(result, Path(markdown_file).with_suffix(".json"))




    ### SELENIUM APPROACH ###
    # knowledge_graph_name = "dbpedia"
    # custom_dataset_papers_dir = os.path.join(custom_dataset_papers_dir, knowledge_graph_name)
    # base_url = f"https://kgqa.github.io/leaderboard/datasets/{knowledge_graph_name}/"
    # wiki_datasets = ["MKQA", "RuBQ-v2", "CronQuestions", "Mintaka", "SimpleQuestionsWikidata", "TimeQuestions - Explicit",
    #             "TimeQuestions - Implicit", "TimeQuestions - Oridinal", "TimeQuestions - Overall",
    #             "TimeQuestions - Temporal Answer", "QALD-9-Plus-Wikidata"]
    # dbpedia_datasets = ["LC-QuAD v1", "LC-QuAD v2", "QALD-1", "QALD-2", "QALD-3", "QALD-4", "QALD-5", "QALD-6", "QALD-7", "QALD-8", "rewordQALD9", "QALD-9"]
    #
    # driver = setup()
    #
    # already_processed_datasets = [dir_name for dir_name in os.listdir(custom_dataset_papers_dir)]
    #
    # # Iterate trough datasets:
    # for dataset in dbpedia_datasets:
    #     if dataset in already_processed_datasets:
    #         logger.info(f"Skipping dataset '{dataset}' as it's already done")
    #         continue
    #
    #     url = base_url + dataset
    #     page_source = get_rendered_page_source(driver, url)
    #     data = extract_relevant_info_from_rendered_page(page_source)
    #
    #     if data:
    #         preprocessed_data = preprocess_data(data, dataset, columns_to_drop, metric_names=known_metrics)
    #
    #         dataset_papers_dir = os.path.join(custom_dataset_papers_dir, dataset)
    #         create_dir_if_not_exists(Path(dataset_papers_dir))
    #
    #         for paper_url in preprocessed_data["PaperUrl"].unique():
    #             logger.info(f"Processing paper '{paper_url}'")
    #             paper_name = Path(paper_url).name
    #             if ".pdf" not in paper_name:
    #                 logger.warning(f"Weird paper name '{paper_name}'")
    #                 paper_name += f".pdf"
    #             download_pdf(paper_url, os.path.join(dataset_papers_dir, paper_name))
    #
    #         result['metric'] = result['metric'].str.lower()
    #         result['Model / System'] = result['Model / System'].str.lower()
    #         result = (
    #             preprocessed_data.groupby("PaperName")
    #             .apply(lambda g: g[["Dataset", "Model / System", "metric", "value"]]
    #                    .rename(columns={
    #                 "Model / System": "Model",
    #                 "metric": "Metric",
    #                 "value": "Result"
    #             })
    #                    .to_dict(orient="records"))
    #             .to_dict()
    #         )
    #         result = create_result_dict_in_correct_format(result)
    #         result = add_hardcoded_task_to_result_dict(result)
    #         save_dict_to_json(result, os.path.join(dataset_papers_dir, "result.json"))
    #
    #     else:
    #         logger.warning(f"Skipping dataset '{dataset}' due to lack of data")
    #
    # driver.quit()
