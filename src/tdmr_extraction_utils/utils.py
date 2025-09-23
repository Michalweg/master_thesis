import pandas as pd
from io import StringIO
from src.tdmr_extraction_utils.data_models import (PreparedTable, TableDecisionResponse, PREPARED_DICT_INTO_STR_TEMPLATE)
from src.logger import logger
from langchain.prompts import PromptTemplate
from src.openai_client import (get_openai_model_response,
                               get_llm_model_response)


def prepare_extracted_tables_for_experiment(
    extracted_tables_dict_object: list[dict],
) -> list[PreparedTable]:
    prepared_dicts = []
    for i, table_object in enumerate(extracted_tables_dict_object):
        csv_data = StringIO(table_object["data"])
        table = pd.read_csv(csv_data)
        table_caption = table_object["caption"]
        table_id = i + 1
        prepared_dict = PreparedTable(
            table=table.to_dict('records'), table_caption=table_caption, table_id=table_id
        )
        prepared_dicts.append(prepared_dict)
    return prepared_dicts


def formate_prepared_dicts_into_str(prepared_dicts: list[PreparedTable]) -> str:
    dicts_into_str = ""

    for prepared_dict in prepared_dicts:
        dicts_into_str += PREPARED_DICT_INTO_STR_TEMPLATE.format(
            table_markdown=prepared_dict.dataframe.head(20).to_markdown(),
            table_caption=prepared_dict.table_caption,
            table_id=prepared_dict.table_id,
        )
        dicts_into_str += "\n"
    return dicts_into_str


def prepare_prompt_for_table_choosing(
    extracted_triplet: dict,
    prepared_dicts_str: str,
    prompt_template: str,
    system_prompt: str = "",

) -> tuple[str, str]:
    if system_prompt:
        prompt = PromptTemplate(
            input_variables=["table_markdown", "table_caption", "table_id"],
            template=prompt_template,
        ).format(triplet=extracted_triplet, tables_data=prepared_dicts_str)
    else:
        prompt_template_combined = system_prompt + prompt_template
        prompt = PromptTemplate(
            input_variables=["table_markdown", "table_caption", "table_id"],
            template=prompt_template_combined,
        ).format(triplet=extracted_triplet, tables_data=prepared_dicts_str)
        system_prompt = ""
    return prompt, system_prompt


def pick_optimal_source_table_for_given_triplet(
    extracted_triplet: dict,
    prepared_dicts: list[PreparedTable],
    prompt_template: str,
    result_object: TableDecisionResponse,
    model_name: str,
    structured_output: bool = True,
    system_prompt: str = "",
) -> dict:
    prepared_dicts_into_str = formate_prepared_dicts_into_str(prepared_dicts)
    prompt, system_prompt = prepare_prompt_for_table_choosing(
        extracted_triplet,
        prepared_dicts_into_str,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
    )
    if structured_output:
        response = get_llm_model_response(prompt=prompt, pydantic_object_structured_output=result_object,
                                          system_prompt=system_prompt, model_name=model_name)
        if response:
            if not isinstance(response, dict):
                response = response.model_dump()
    else:
        response = get_llm_model_response(
            prompt=prompt, pydantic_object_structured_output=None,
            system_prompt=system_prompt, model_name=model_name
        )
    logger.info(f"Response for picking table: {response}")
    return response