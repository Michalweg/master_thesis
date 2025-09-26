from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from pathlib import Path
import re
import pandas as pd
import json
from typing import Union
from io import StringIO
from tqdm import tqdm
import os
from src.utils import save_str_as_markdown, read_markdown_file_into_lines, extract_tables_from_markdown, saving_list_of_dfs, create_dir_if_not_exists, save_dict_to_json

table_caption_prompt_template = """
You are given output from a mechanism which aims to extract tables captions. This mechanism make sometimes mistakes, so 
thus tou need to correct it if necessary. You will be given an extracted caption by this mechanism as well as surrounding text 
from the paper, from which the table caption itself was extracted. Your task is to analyze the provided text and output the correct 
table caption. Do not modify the original text, just output correct table caption from the provided context. 
Please output only the correct table caption with no explanation without any other comments or explanations. 
Please also make sure that in case of table captions being presented in the text extracted as table caption and also surrounding text, please output the first option (the text extracted as table caption). 
DO NOT MODIFY the text related to caption, include also details like table name (if present in the text).

Here is tbe text which was extracted as table caption: 
{table_caption}

Here is the text before extracted caption:
{text_before} 

Here is the text after  extracted caption:
{text_after}
"""


def send_request_to_the_model_with_ollama(prompt: str,
                                          model_name: str,
                                          url: str = "http://localhost:11434/api/generate") -> str:

    import subprocess
    data = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0
    })
    # Curl command split into a list
    curl_command = [
        "curl",
        "http://localhost:11434/api/generate",
        "-d", data,
        "-H", "Content-Type: application/json"
    ]
    # Run the curl command
    result = subprocess.run(curl_command, text=True, capture_output=True)
    r = json.loads(result.stdout)['response']

    # r = requests.post(url, json=payload)
    #
    # # Parsing and saving LLM response
    # r = json.loads(r.content)["response"]

    return r


def find_extracted_texts_given_cref(extracted_texts: list, cref: str) -> str:
    for text_element in extracted_texts:
        if text_element.self_ref == cref:
            return text_element.text
    return ''


def setup_document_converter():
    """
    Initializes and returns a DocumentConverter instance with specified options.
    """
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return doc_converter


def convert_pdf_into_md_using_docling(paper_path: Path) -> str:
    converter = setup_document_converter()
    converted_document = converter.convert(paper_path)
    with open(paper_path.with_suffix(".md"), "w") as f:
        f.write(converted_document.document.export_to_markdown())
    return str(paper_path.with_suffix(".md"))




if __name__ == "__main__":
    doc_converter =setup_document_converter()


    papers_dir =  'custom_dataset_papers/CronQuestions'
    results_dir = "custom_dataset_papers/CronQuestions" # results
    create_dir_if_not_exists(Path(results_dir))
    already_processed_files = [
            paper_path.name for paper_path in Path(results_dir).iterdir()
        ]

    files_list = list(Path(papers_dir).iterdir())

    for source in tqdm(files_list):

        if source.stem in already_processed_files:
            print(f"File has been already processed: {source.name}")
            continue

        if source.suffix == '.pdf':
            paper_name = source.stem
            create_dir_if_not_exists(Path(os.path.join(results_dir, paper_name)))
            tables_captions_list = []
            result = doc_converter.convert(source)
            extracted_texts = result.document.texts

            with open(source.stem + ".md", "w") as f:
                f.write(result.document.export_to_markdown())


            for table_ix, table in enumerate(result.document.tables):
                table_caption_str = ''
                table_df: pd.DataFrame = table.export_to_dataframe()

                for table_caption in table.captions:
                    extracted_table_caption = find_extracted_texts_given_cref(extracted_texts, table_caption.cref)
                    print("\n")
                    print(f"Extracted caption: {extracted_table_caption}")
                    # extracted_caption_cref_str, extracted_caption_cref_id = "/".join(table_caption.cref.split("/")[:-1]), int(table_caption.cref.split("/")[-1])
                    # text_before_caption = find_extracted_texts_given_cref(extracted_texts, extracted_caption_cref_str + "/" + str(extracted_caption_cref_id - 1))
                    # text_after_caption = find_extracted_texts_given_cref(extracted_texts, extracted_caption_cref_str + "/" + str(extracted_caption_cref_id + 1))
                    # correct_caption_prompt = table_caption_prompt_template.format(table_caption=extracted_table_caption,text_before=text_before_caption,text_after=text_after_caption)
                    # correct_catpion_extracted = send_request_to_the_model_with_ollama(correct_caption_prompt, "llama3.2")
                    # print(f"prompt to the model: {correct_caption_prompt}")
                    # print(f"Caption extract by the model: {correct_catpion_extracted}")
                    # print("*"*200)
                    table_caption_str += extracted_table_caption

                csv_buffer = StringIO()
                table_df.to_csv(csv_buffer, index=False)
                tables_captions_list.append({"caption": table_caption_str,
                                             "data":  csv_buffer.getvalue()})
                table_df.to_csv(os.path.join(results_dir, paper_name, str(table_ix) + ".csv"))


            save_dict_to_json(tables_captions_list, os.path.join(results_dir, paper_name,"result_dict.json"))
