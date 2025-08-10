"""
In this file the openai client code is put and also extraction of triplets from entire FILE.
"""

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.beta.threads.message_create_params import (
    Attachment, AttachmentToolFileSearch)
from openai.types.responses import ParsedResponse
from pydantic import BaseModel, Field
import os

from prompts.triplets_extraction import triplets_extraction_prompt_gpt_4_turbo, triplets_extraction_prompt_gpt_4_turbo_more_context, triplets_extraction_prompt_gpt_4o, openai_gpt_oss_120b_system_prompt, openai_gpt_oss_120b_user_prompt
from src.triplets.triplets_unification import (
    extract_unique_triplets_from_normalized_triplet_file,
    normalize_strings_triplets)
from src.utils import (create_dir_if_not_exists, read_json, save_dict_to_json,
                       save_str_as_txt_file)

load_dotenv()
from src.logger import logger
from src.utils import count_tokens_in_prompt
from typing import Union, List
from src.const import OPENAI_API_MODELS

MAXIMUM_MO_TOKENS_PER_PROMPT = 10_000
MODEL_NAME = "openai-gpt-oss-120b"

client = OpenAI()

class TooManyTokensError(Exception):
    """Raised when the prompt exceeds the maximum allowed token count."""
    def __init__(self, token_count: int, max_allowed: int):
        self.token_count = token_count
        super().__init__(f"Prompt has {token_count} tokens, exceeds limit of {max_allowed}.")


class ValidTriplet(BaseModel):
    Dataset: str = Field(
        description="The specific machine learning problem being solved."
    )
    Metric: str = Field(
        description="The specific, named dataset used for evaluation."
    )
    Task: str = Field(description=" The quantitative metric used to report the result.")


class ExtractedTriplets(BaseModel):
    extracted_triplets: list[ValidTriplet] = Field(description="List of complete, containing all fields triplets containing data regarding machine learning experiments from the given part of a research paper. ")


def setup_academic_client():
    api_key = os.getenv("ACADEMIC_API_KEY") # Replace with your API key
    base_url = "https://chat-ai.academiccloud.de/v1"
    return OpenAI(api_key=api_key, base_url=base_url)


def get_open_model_response(prompt: str, model_name: str, system_prompt: str, pydantic_object_structured_output: type[BaseModel]):
    client = setup_academic_client()
    if pydantic_object_structured_output:
        try:
            chat_completion = client.responses.parse(
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
                text_format=pydantic_object_structured_output,
            )
            return chat_completion.output_parsed
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}, skipping this triplet")
    else:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt if system_prompt else "You are a helpful assistant"},
                      {"role": "user", "content": prompt}],
            model=model_name,
        )
        return chat_completion.choices[0].message.content

def get_llm_model_response(prompt: str, model_name: str, system_prompt: str = "", pydantic_object_structured_output: type[BaseModel] = None) -> Union[str, type[BaseModel]]:
    if model_name in OPENAI_API_MODELS:
        return get_openai_model_response(prompt, model_name, system_prompt, pydantic_object_structured_output)
    else:
        return get_open_model_response(prompt, model_name, system_prompt, pydantic_object_structured_output)


def get_openai_model_response(
    prompt: str, model_name: str = "gpt-4o", system_prompt: str = "", pydantic_object_structured_output: type[BaseModel] = None
):
    no_of_token_in_prompt = 10 # count_tokens_in_prompt(prompt, model_name)
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
    prompt: str, pydantic_object: type[BaseModel], model_name: str = "gpt-4o", system_prompt="", client=client
) -> ParsedResponse[BaseModel]:
    no_of_token_in_prompt = count_tokens_in_prompt(prompt, model_name)
    if no_of_token_in_prompt > MAXIMUM_MO_TOKENS_PER_PROMPT:
        logger.error(
            f"The prompt you are about to send is too large: {no_of_token_in_prompt}"
        )
        raise TooManyTokensError(token_count=no_of_token_in_prompt, max_allowed=MAXIMUM_MO_TOKENS_PER_PROMPT)
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
        return response
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

    prompt_without_file_id = triplets_extraction_prompt_gpt_4_turbo_more_context

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
    # papers_dir = "leaderboard-generation-papers"
    # test_papers = ['2020.findings-emnlp.378', '1905.12598', '1809.08370']
    # papers_dir_list = list(Path(papers_dir).iterdir())
    # papers_dir_list = [x for x in papers_dir_list if x.stem in test_papers]
    # output_dir = (
    #     f"triplets_extraction/from_entire_document_refined_prompt_{MODEL_NAME}_13_07_update"
    # )
    # create_dir_if_not_exists(Path(output_dir))
    # already_processed_files = [
    #     paper_path.name for paper_path in Path(output_dir).iterdir()
    # ]
    #
    # import os
    #
    # from langchain_core.output_parsers import JsonOutputParser
    #
    # output_parser = JsonOutputParser()
    #
    # for i, paper_path in tqdm(enumerate(papers_dir_list)):
    #     print("Working on: {}".format(paper_path))
    #     dir_to_save_paper_results = ""
    #
    #     if paper_path.suffix == ".pdf":
    #         paper_name = paper_path.stem
    #         try:
    #             inputpdf = PdfReader(open(paper_path, "rb"))
    #         except Exception as e:
    #             print(f"Skipping this file: {paper_path}")
    #             continue
    #
    #         num_pages = len(inputpdf.pages)
    #
    #         if paper_path.stem in already_processed_files:
    #             print(f"This file: {paper_path.stem} has already been processed")
    #             continue
    #         dir_to_save_paper_results = os.path.join(output_dir, paper_name)
    #         create_dir_if_not_exists(Path(dir_to_save_paper_results))
    #         # Split PDF to each of two subpages
    #         # Process the PDF in chunks of two pages
    #         for i in range(0, num_pages, 2):
    #             output = PdfWriter()
    #             # Add the current page
    #             output.add_page(inputpdf.pages[i])
    #             # Add the next page if it exists
    #             if i + 1 < num_pages:
    #                 output.add_page(inputpdf.pages[i + 1])
    #
    #             # Create a temporary file for storing the two-page PDF
    #             # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
    #             temp_file_path = "temp_file.pdf"
    #             with open(temp_file_path, "wb") as temp_file:
    #                 output.write(temp_file)
    #             # temp_file_path = temp_file.name  # Store the temp file path
    #             model_response = get_openai_model_response_based_on_the_whole_document(
    #                 model_name=MODEL_NAME, file_path=temp_file_path
    #             )
    #             if model_response:
    #                 try:
    #                     parsed_response = output_parser.parse(model_response)
    #                     if parsed_response:
    #                         save_dict_to_json(
    #                             parsed_response,
    #                             os.path.join(
    #                                 dir_to_save_paper_results,
    #                                 paper_path.stem + f"_{str(i)}.json",
    #                             ),
    #                         )
    #                     else:
    #                         print(f"Parsed response is empty")
    #                 except:
    #                     print(
    #                         f"There was problem with parsing this passage: {model_response}"
    #                     )
    #                     save_str_as_txt_file(
    #                         os.path.join(
    #                             dir_to_save_paper_results, paper_path.stem + ".txt"
    #                         ),
    #                         model_response,
    #                     )
    #             else:
    #                 print(
    #                     f"From the {i} two pages it cannot read any of the tables from this {paper_name} paper!"
    #                 )
    #             os.remove(temp_file_path)
    #     combine_and_unique_triplets_for_a_given_paper(dir_to_save_paper_results)
    test_papers = ['2020.findings-emnlp.378', '1905.12598', '1809.08370']
    from src.utils import read_markdown_file_content
    output_dir = (
        f"triplets_extraction/from_entire_document_refined_prompt_{MODEL_NAME}_09_08_update"
    )
    for f in Path("research-papers-markdwons").iterdir():
        if not f.stem in test_papers:
            continue

        if f.stem != "1809.08370":
            continue

        if f.suffix == ".md" and "read" not in f.stem.lower():
            dir_to_save_paper_results = os.path.join(output_dir, f.stem)
            create_dir_if_not_exists(Path(dir_to_save_paper_results))
            file_content = read_markdown_file_content(f)
            chunk_size = 5000
            valid_triplets_list = []
            valid_jsons_list = []
            for i in range(0, len(file_content), chunk_size):
                chunk = file_content[i:i + chunk_size]
                model_response = get_llm_model_response(prompt=openai_gpt_oss_120b_user_prompt.format(research_paper=chunk),
                                                        model_name=MODEL_NAME, system_prompt=openai_gpt_oss_120b_system_prompt,
                                                        pydantic_object_structured_output=ExtractedTriplets)
                if model_response:
                    model_response = model_response.extracted_triplets
                    if model_response:
                        logger.info(f"The extracted triplets for the chunk: {model_response}")
                        valid_triplets_list.append(model_response)

            for output_object in valid_triplets_list:
                if isinstance(output_object, list):
                    valid_jsons_list.extend([x.model_dump() for x in output_object])
                else:
                    valid_jsons_list.append(output_object.model_dump())

            normalized_strings_triplets = normalize_strings_triplets(valid_jsons_list)
            unique_triplets = extract_unique_triplets_from_normalized_triplet_file(
                normalized_strings_triplets
            )
            save_dict_to_json(
                unique_triplets,
                os.path.join(
                    dir_to_save_paper_results,
                    "unique_triplets.json"
                ),
            )