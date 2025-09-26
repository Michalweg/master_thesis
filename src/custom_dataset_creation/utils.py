from jupyterlab.semver import valid
from pydantic import BaseModel, Field
from src.utils import read_json, save_dict_to_json
from collections import defaultdict
from pathlib import Path
from src.triplets.triplets_unification import deduplicate_outputs, extract_unique_triplets_from_normalized_triplet_file, normalize_strings_triplets
import os

class ExtendedTuple(BaseModel):
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
    Model: str = Field(
        description="A model that obtained given result for the specific metric on the given task. "
    )


class GroundTruthTuple(BaseModel):
    PaperUrl: str
    TDMs: list[dict]


def parse_notebook_lm_output_into_ground_truth_structure(notebook_lm_outputs: list[dict]) -> dict[str, GroundTruthTuple]:
    ground_truth_structured_output = defaultdict(GroundTruthTuple)
    for notebook_lm_output in notebook_lm_outputs:
        if notebook_lm_output["PaperName"] in ground_truth_structured_output:
            ground_truth_structured_output[notebook_lm_output["PaperName"]].TDMs.append({
                "Task": notebook_lm_output["Task"],
                "Dataset": notebook_lm_output["Dataset"],
                "Metric": notebook_lm_output["Metric"],
            })
        else:
            ground_truth_structured_output[notebook_lm_output["PaperName"]] = GroundTruthTuple(PaperUrl="",
                                                                                               TDMs=[{"Task": notebook_lm_output["Task"],
                                                                                                      "Dataset": notebook_lm_output["Dataset"],
                                                                                                      "Metric": notebook_lm_output["Metric"]}])
    return ground_truth_structured_output

def prepare_extracted_triplets(ground_truth_triplets: dict[str, dict]) -> dict[str, dict]:
    for paper_name in ground_truth_triplets:
        initial_triplets = ground_truth_triplets[paper_name]["TDMs"]
        normalized_triplets = normalize_strings_triplets(initial_triplets)
        deduplicated_triplets = deduplicate_outputs(normalized_triplets)
        unique_triplets = extract_unique_triplets_from_normalized_triplet_file(deduplicated_triplets)
        ground_truth_triplets[paper_name]["TDMs"] = unique_triplets

    return ground_truth_triplets


if __name__ == "__main__":
    notebook_lm_result_file_path = "custom_dataset_papers/result-notebook-lm-more-general.json"
    notebook_lm_result_file = read_json(notebook_lm_result_file_path)
    valid_structured_output = parse_notebook_lm_output_into_ground_truth_structure(notebook_lm_result_file)
    valid_structured_output = {paper_name: valid_structured_output[paper_name].model_dump() for paper_name in valid_structured_output}
    validated_structured_output = prepare_extracted_triplets(valid_structured_output)
    save_dict_to_json(validated_structured_output, os.path.join(Path(notebook_lm_result_file_path).parent, Path(notebook_lm_result_file_path).stem + "_ground_truth.json"))