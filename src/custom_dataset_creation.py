import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from src.utils import extract_tables_from_markdown


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

    if tabulator_table:
        print(tabulator_table.prettify())
    else:
        print("Tabulator element not found.")

    rows = []

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
    data: list[dict], dataset_name: str, columns_to_drop: list[str]
) -> pd.DataFrame:
    df = pd.DataFrame.from_records(data)
    df["Dataset"] = dataset_name
    columns_to_drop_in_df = set(df.columns).intersection(columns_to_drop)
    df = df.drop(columns=list(columns_to_drop_in_df), axis=1)
    return df


if __name__ == "__main__":
    mardkown_file = "custom_dataset_inputs/qald.md"
    tabels = extract_tables_from_markdown(mardkown_file)
    print(tabels)
    url = "https://kgqa.github.io/leaderboard/datasets/wikidata/MKQA"
    base_url = "https://kgqa.github.io/leaderboard/datasets/wikidata/"
    datasets = ["MKQA", "RuBQ-v2"]
    columns_to_drop = ["Year", "Language", "Reported by", "id"]
    known_metrics = ["F1", "Precision", "Recall"]
    driver = setup()

    # Iterate trough datasets:
    for dataset in datasets:
        url = base_url + dataset
        page_source = get_rendered_page_source(driver, url)
        data = extract_relevant_info_from_rendered_page(page_source)
        preprocessed_data = preprocess_data(data, dataset, columns_to_drop)
        print(preprocessed_data)

    driver.quit()
