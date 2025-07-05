"""
In this file the openai client code is put and also extraction of triplets from entire FILE
"""

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.beta.threads.message_create_params import (
    Attachment, AttachmentToolFileSearch)
from pydantic import BaseModel
from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm

from prompts.triplets_extraction import triplets_extraction_prompt_gpt_4_turbo
from src.triplets.triplets_unification import (
    extract_unique_triplets_from_normalized_triplet_file,
    normalize_strings_triplets)
from src.utils import (create_dir_if_not_exists, read_json, save_dict_to_json,
                       save_str_as_txt_file)

load_dotenv()
from src.logger import logger
from src.utils import count_tokens_in_prompt

MAXIMUM_MO_TOKENS_PER_PROMPT = 10_000
MODEL_NAME = "gpt-4-turbo"

client = OpenAI()


def get_openai_model_response(
    prompt: str, model_name: str = "gpt-4o", system_prompt: str = ""
):
    no_of_token_in_prompt = count_tokens_in_prompt(prompt, model_name)
    if no_of_token_in_prompt > MAXIMUM_MO_TOKENS_PER_PROMPT:
        print(f"The prompt you are about to send is too large: {no_of_token_in_prompt}")
        raise ValueError
    try:
        # Send a request to the GPT-4 model
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant."
                        if not system_prompt
                        else system_prompt
                    ),
                },  # System message to set behavior
                {"role": "user", "content": prompt},  # User's prompt
            ],
            temperature=0,  # Adjust for creativity (0 = deterministic, 1 = very creative)
            max_tokens=1000,  # Adjust for the length of the response
        )
        # Extract the content of the assistant's reply
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"


def get_openai_model_structured_response(
    prompt: str, pydantic_object, model_name: str = "gpt-4o", system_prompt=""
) -> BaseModel:
    no_of_token_in_prompt = count_tokens_in_prompt(prompt, model_name)
    if no_of_token_in_prompt > MAXIMUM_MO_TOKENS_PER_PROMPT:
        logger.error(
            f"The prompt you are about to send is too large: {no_of_token_in_prompt}"
        )
        return None
    try:
        response = client.responses.parse(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        if system_prompt
                        else "You are a helpful assistant."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            text_format=pydantic_object,
        )
        return response.output_parsed
    except Exception as e:
        print(f"An error occurred: {e}")
        raise ValueError


def get_openai_model_response_based_on_the_whole_document(
    model_name: str, file_path: str
):
    file = client.files.create(file=open(file_path, "rb"), purpose="assistants")

    pdf_assistant = client.beta.assistants.create(
        model=model_name,
        description="An assistant to extract the contents of PDF files.",
        tools=[{"type": "file_search"}],
        name="PDF assistant",
    )

    # Create thread
    thread = client.beta.threads.create()

    prompt_without_file_id = triplets_extraction_prompt_gpt_4_turbo

    # Create assistant
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        attachments=[
            Attachment(
                file_id=file.id, tools=[AttachmentToolFileSearch(type="file_search")]
            )
        ],
        content=prompt_without_file_id,
    )

    # Run thread
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=pdf_assistant.id, timeout=1000
    )

    if run.status != "completed":
        raise Exception("Run failed:", run.status)

    messages_cursor = client.beta.threads.messages.list(thread_id=thread.id)
    messages = [message for message in messages_cursor]

    # Output text
    res_txt = messages[0].content[0].text.value
    print(res_txt)

    # return completion.choices[0].message.content
    return res_txt


def combine_and_unique_triplets_for_a_given_paper(paper_triplets_dir: str):
    if paper_triplets_dir:
        all_extracted_triplets = []
        for file in os.listdir(paper_triplets_dir):
            if file.endswith(".json"):
                all_extracted_triplets.extend(
                    read_json(Path(os.path.join(paper_triplets_dir, file)))
                )
        normalized_strings_triplets = normalize_strings_triplets(all_extracted_triplets)
        unique_triplets = extract_unique_triplets_from_normalized_triplet_file(
            normalized_strings_triplets
        )
        save_dict_to_json(
            unique_triplets, Path(f"{paper_triplets_dir}/unique_triplets.json")
        )


if __name__ == "__main__":
    papers_dir = "leaderboard-generation-papers"
    papers_dir_list = list(Path(papers_dir).iterdir())
    output_dir = (
        f"triplets_extraction/from_entire_document_refined_prompt_{MODEL_NAME}_gemini"
    )
    create_dir_if_not_exists(Path(output_dir))
    already_processed_files = [
        paper_path.name for paper_path in Path(output_dir).iterdir()
    ]

    import os

    from langchain_core.output_parsers import JsonOutputParser

    output_parser = JsonOutputParser()

    for i, paper_path in tqdm(enumerate(papers_dir_list)):
        print("Working on: {}".format(paper_path))
        dir_to_save_paper_results = ""

        if paper_path.suffix == ".pdf":
            paper_name = paper_path.stem
            try:
                inputpdf = PdfReader(open(paper_path, "rb"))
            except Exception as e:
                print(f"Skipping this file: {paper_path}")
                continue

            num_pages = len(inputpdf.pages)

            if paper_path.stem in already_processed_files:
                print(f"This file: {paper_path.stem} has already been processed")
                continue
            dir_to_save_paper_results = os.path.join(output_dir, paper_name)
            create_dir_if_not_exists(Path(dir_to_save_paper_results))
            # Split PDF to each of two subpages
            # Process the PDF in chunks of two pages
            for i in range(0, num_pages, 2):
                output = PdfWriter()
                # Add the current page
                output.add_page(inputpdf.pages[i])
                # Add the next page if it exists
                if i + 1 < num_pages:
                    output.add_page(inputpdf.pages[i + 1])

                # Create a temporary file for storing the two-page PDF
                # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file_path = "temp_file.pdf"
                with open(temp_file_path, "wb") as temp_file:
                    output.write(temp_file)
                # temp_file_path = temp_file.name  # Store the temp file path
                model_response = get_openai_model_response_based_on_the_whole_document(
                    model_name=MODEL_NAME, file_path=temp_file_path
                )
                if model_response:
                    try:
                        parsed_response = output_parser.parse(model_response)
                        if parsed_response:
                            save_dict_to_json(
                                parsed_response,
                                os.path.join(
                                    dir_to_save_paper_results,
                                    paper_path.stem + f"_{str(i)}.json",
                                ),
                            )
                        else:
                            print(f"Parsed response is empty")
                    except:
                        print(
                            f"There was problem with parsing this passage: {model_response}"
                        )
                        save_str_as_txt_file(
                            os.path.join(
                                dir_to_save_paper_results, paper_path.stem + ".txt"
                            ),
                            model_response,
                        )
                else:
                    print(
                        f"From the {i} two pages it cannot read any of the tables from this {paper_name} paper!"
                    )
                os.remove(temp_file_path)
        combine_and_unique_triplets_for_a_given_paper(dir_to_save_paper_results)
