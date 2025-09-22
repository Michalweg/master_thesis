import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional

class TableDecisionResponse(BaseModel):
    explanation: str = Field(
        description="A reasoning behind choosing a table to be used to extract result for a given triplet."
    )
    table_id_to_extract_result_metric_from: int = Field(
        description="A table id which should be used to extract result for a given triplet."
    )

class ExtendedTDMRWithModel(BaseModel):
    Task: str = Field(
        description="A task from the triplet for which result is obtained"
    )
    Metric: str = Field(
        description="A metric from the triplet for which result is obtained"
    )
    Dataset: str = Field(
        description="A dataset from the triplet for which result is obtained"
    )
    Result: float = Field(
        description="A result for given triplet (task, dataset, metric) extracted from the provided table. Output only the value!"
    )
    Model: str = Field(description="A model designed by the authors from the triplet for which result is obtained.")

class TdmrExtractionResponseWithModel(BaseModel):
    explanation: str = Field(description="A reasoning why the given result was obtained or why the second part of the output is None. ")
    tdmr_output: ExtendedTDMRWithModel | None = Field("A extracted tdmr_output or None in case of not finding relevant data within the provided table. ")

class TdmrExtractionResponseSplit(BaseModel):
    explanation: str = Field(description="A reasoning why the given result was obtained.")
    Task: str = Field(
        description="A task from the triplet for which result is obtained"
    )
    Metric: str = Field(
        description="A metric from the triplet for which result is obtained"
    )
    Dataset: str = Field(
        description="A dataset from the triplet for which result is obtained"
    )
    Result: float = Field(
        description="A result for given triplet (task, dataset, metric) extracted from the provided table. Output only the value!"
    )

class PreparedTable(BaseModel):
    # Store the table data as a list of dictionaries (records)
    table: List[Dict[str, Any]] = Field(description="Table data in a records format (list of dicts)")
    table_caption: str = Field(description="Table caption for the provided table")
    table_id: int = Field(description="A table id of the provided table")

    # Optional: Add a property to access the DataFrame on demand
    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns the table data as a pandas DataFrame."""
        return pd.DataFrame.from_records(self.table)


PREPARED_DICT_INTO_STR_TEMPLATE = """
Table_markdown: {table_markdown}
Table_caption: {table_caption}
Table_id: {table_id}
"""