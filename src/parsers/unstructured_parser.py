import os
from pathlib import Path

import pandas as pd
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf

from src.parsers.llm_parser import send_request_to_the_model_with_ollama
from src.utils import create_dir_if_not_exists, save_dict_to_json

os.environ["OCR_AGENT"] = "tesseract"

MODEL_NAME = "gpt-4o"
MODEL_NAME_FOR_CAPTION_ASSIGNMENT = "llama3.1"

PROMPT_FOR_ASSIGNING_TEXT_AND_CAPTION = """
Human: You will receive three inputs:
	1.	A table in .csv format (structured data).
	2.	A textual description of a table (table text).
	3.	A proposed caption for the table.

Your task is to determine if the provided table text and caption accurately describe the table presented in the .csv format. Evaluate based on the content, structure, and semantics of the data in the .csv file.

If the table text and caption match the table in the .csv file, respond with True. If they do not match, respond with False. Provide no additional explanations or comments.

Inputs:
	•	Table (in .csv format): [Insert table data here]
	•	Table text: [Insert table text here]
	•	Table caption: [Insert table caption here]

Output:
	•	True or False

Here is the table in .csv format: 
{csv_table} 

Here is the table text: 
{text_table}

Here is the table caption:
{table_caption}

Assistant:
"""


def assign_table_text_and_caption_to_a_table(
    table_text_captions_dict: dict, papers_tables_dir: str
):
    papers_tables_dir = list(Path(papers_tables_dir).iterdir())
    for paper_table_path in papers_tables_dir:
        # paper_table = pd.read_csv(paper_table_path)
        papers_tables_from_dict = table_text_captions_dict["text"]
        tables_captions = table_text_captions_dict["captions"]
        tables_htmls = table_text_captions_dict["table_html"]
        for index in range(len(tables_captions)):
            table_text, table_caption = (
                papers_tables_from_dict[index],
                tables_captions[index],
            )
            prompt = PromptTemplate(
                template=PROMPT_FOR_ASSIGNING_TEXT_AND_CAPTION
            ).format(
                text_table=table_text,
                table_caption=table_caption,
                csv_table=pd.read_html(tables_htmls[index]),
            )
            model_output = send_request_to_the_model_with_ollama(
                prompt, model_name=MODEL_NAME_FOR_CAPTION_ASSIGNMENT, url=""
            )
            if model_output == "True":
                table_text_captions_dict.update({"table": paper_table_path})
                return table_text_captions_dict


if __name__ == "__main__":
    papers_dir = "various_tables_extraction_approaches_paper_dir"
    experiment_dir = f"various_tables_extraction_approaches_summary_dir"
    papers_dir_list = list(Path(papers_dir).iterdir())
    tables_text_with_captions_dir = "tables_with_captions_dir_unstructured"
    create_dir_if_not_exists(Path(tables_text_with_captions_dir))

    for paper_path in (pbar := tqdm(papers_dir_list)):
        pbar.set_description(f"Processing document: {paper_path.name}")
        print("\n\n")

        # Creation of all necessary variables
        text_by_page = {}
        tables = []
        captions = []
        tables_htmls = []
        tables_and_captions_dict = {"text": "", "captions": [], "table_html": ""}

        if paper_path.suffix == ".pdf":
            elements = partition_pdf(
                paper_path, strategy="hi_res", infer_table_structure=True
            )
            for i, el in enumerate(elements):
                if el.category == "Table":
                    try:
                        if (elements[i - 1].text not in captions) and (
                            (elements[i - 1].category == "FigureCaption")
                            or (elements[i - 1].text[:5] == "Table")
                        ):

                            tables.append(elements[i - 1].text + "\n" + el.text)
                            tables_htmls.append(el.metadata.text_as_html)
                            captions.append(elements[i - 1].text)

                        elif (elements[i + 1].text not in captions) and (
                            (elements[i + 1].category == "FigureCaption")
                            or (elements[i + 1].text[:5] == "Table")
                        ):
                            tables_htmls.append(el.metadata.text_as_html)

                            tables.append(elements[i + 1].text + "\n" + el.text)
                            captions.append(elements[i + 1].text)

                        # else:
                        #     tables.append(el.text)
                        #     tables_htmls.append(el.metadata.text_as_html)
                    except IndexError:
                        continue
                        # tables.append(el.text)

            tables_and_captions_dict["text"] = tables
            tables_and_captions_dict["captions"] = captions
            tables_and_captions_dict["table_html"] = tables_htmls
            save_dict_to_json(
                tables_and_captions_dict,
                os.path.join(
                    tables_text_with_captions_dir,
                    f"{paper_path.stem}_tables_with_captions.json",
                ),
            )
            # assign_table_text_and_caption_to_a_table(tables_and_captions_dict, os.path.join(experiment_dir, paper_path.stem, 'manual_extracted_tables'))
            # tables = [el for el in elements if el.category == "Table"]
            #
            # print(tables[0].text)
            # print(tables[0].metadata.text_as_html)
