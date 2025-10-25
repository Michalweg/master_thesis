import json
import os
import re
import subprocess
from pathlib import Path
from typing import Union

import Levenshtein
import pandas as pd
import requests
import tiktoken
from dotenv import load_dotenv

from src.logger import logger


def set_env():
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    load_dotenv("../.env")


def save_data_to_json_file(
    extracted_text_dict: dict, output_dir_path: str | Path
) -> None:
    logger.info("Saving json with extracted text into: {}".format(output_dir_path))
    output_dir_path = (
        output_dir_path if isinstance(output_dir_path, Path) else Path(output_dir_path)
    )
    save_dict_to_json(extracted_text_dict, output_dir_path / "extracted_text_dict.json")


def saving_list_of_dfs(dataframes: list, output_dir_path: Path) -> None:
    logger.info("Saving dataframes into: {}".format(output_dir_path))
    for i, df in enumerate(dataframes):
        df.to_csv(output_dir_path / f"df_{str(i)}.csv", index=False)
    logger.info(
        "Saving json with extracted text into: {} complete".format(output_dir_path)
    )


def save_dict_to_json(data: Union[dict, list], file_path: str | Path) -> None:
    # Ensure the input is a dictionary
    if isinstance(data, dict):
        # Open a file in write mode and dump the dictionary into it as JSON
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"JSON file '{file_path}' created successfully.")
    elif isinstance(data, list):
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"JSON file '{file_path}' created successfully.")
    else:
        print("The provided data is not a dictionary nor a list.")


def run_bash_command(command: str) -> str:
    """Run a bash command and print its output and errors."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        print("Error occurred while executing command:\n", e)
        return ""


def create_dir_if_not_exists(dir_path: Path) -> None:
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def read_markdown_file_into_lines(markdown_file_path: Path | str) -> list:
    with open(markdown_file_path, "r") as file:
        lines = file.readlines()

    return lines


def read_markdown_file_content(markdown_file_path: Path | str) -> str:
    with open(markdown_file_path) as markdown_file:
        files_content = markdown_file.read()
    return files_content


def extract_tables_from_markdown(md_file_path: str) -> list[pd.DataFrame]:
    lines = read_markdown_file_into_lines(md_file_path)

    tables = []
    table_lines = []
    inside_table = False

    for line in lines:
        # Check if the line is part of a table (contains '|' and isn't part of a code block or text)
        if re.match(r"^\|.+\|$", line.strip()):  # Detects lines with table content
            table_lines.append(line.strip())
            inside_table = True
        elif inside_table and not re.match(
            r"^\|.+\|$", line.strip()
        ):  # Exit when table ends
            inside_table = False
            if table_lines:
                tables.append(table_lines)
                table_lines = []  # Reset for the next table

    if not tables:
        return []  # No table found

    dataframes = []
    for table in tables:
        header = table[0].split("|")[
            1:-1
        ]  # Remove first and last empty strings due to '|'
        rows = [
            line.split("|")[1:-1] for line in table[2:]
        ]  # Skip the separator line (index 1)

        # Process each cell to remove commas between digits
        for i, row in enumerate(rows):
            rows[i] = [re.sub(r"(?<=\d),(?=\d)", "", cell.strip()) for cell in row]
        try:
            df = pd.DataFrame(rows, columns=header)
            df.columns = [col.strip() for col in df.columns]
            dataframes.append(df)
        except:
            logger.error("")

    return dataframes


def remove_table_data_from_markdown(
    markdown_file_path: str, markdown_without_tables_file_path: str
) -> None:
    lines = read_markdown_file_into_lines(markdown_file_path)

    markdown_file_without_table = ""

    for i, line in enumerate(lines):
        if re.match(r"^\|.+\|$", line.strip()):
            continue
        else:
            markdown_file_without_table += line

    save_str_as_markdown(markdown_without_tables_file_path, markdown_file_without_table)


def find_closest_string(target: str, string_list: list) -> str:
    closest_string = None
    min_distance = float("inf")
    target = target.lower()
    for s in string_list:
        distance = Levenshtein.distance(target, s.lower())
        if distance < min_distance:
            min_distance = distance
            closest_string = s

    return closest_string


def save_str_as_markdown(marker_markdown_file_path, file_content) -> None:
    with open(marker_markdown_file_path, "w") as markdown_file:
        markdown_file.write(file_content)


def save_str_as_txt_file(txt_file_path, str_content) -> None:
    with open(txt_file_path, "w") as txt_file:
        txt_file.write(str_content)


def read_json(json_file_path: Union[Path|str]) -> Union[dict | list[dict]]:
    if Path(json_file_path).exists():
        with open(json_file_path, "r") as json_file:
            return json.load(json_file)
    else:
        logger.warning(f"The provided json file '{json_file_path}' does not exist.")
        return {}


def count_tokens_in_prompt(prompt: str, model_name: str) -> int:
    model_encoding_dict = {
        "gpt-4o": "o200k_base",
        "gpt-4": "cl100k_base",
        "gpt3.5": "cl100k_base",
    }
    if model_name not in model_encoding_dict:
        raise KeyError(f"Provided model is not supported")
    encoding = tiktoken.get_encoding(model_encoding_dict[model_name])
    return len(encoding.encode(prompt))


def get_unique_values_with_the_same_order(data: list) -> list:
    used = set()
    unique = [
        element
        for element in data
        if element not in used and (used.add(element) or True)
    ]
    return unique


def download_pdf(url, filename):
    """
    Download a PDF from a given URL and save it with the specified filename.

    Args:
        url (str): The URL of the PDF to download.
        filename (str): The local filename to save the PDF as.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(filename, "wb") as f:
            f.write(response.content)
        logger.info(f"Downloaded PDF as '{filename}'")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF: {e} for url; {url}")


if __name__ == "__main__":
    result_fle_nam = "result-notebook-lm.json"
    dir_path = "custom_dataset_papers"
    tasks_set = set()
    for dataset_name in Path(dir_path).iterdir():
        if dataset_name.is_dir():
            file_path = dataset_name.joinpath(result_fle_nam)
            try:
                result = read_json(file_path)
            except:
                continue

            for item in result:
                tasks_set.add(item["Task"])

    print(tasks_set)