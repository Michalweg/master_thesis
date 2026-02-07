import os
from pathlib import Path

from langchain.prompts import PromptTemplate
from tqdm import tqdm

from src.openai_client import get_llm_model_response
from src.triplets.triplets_unification import (
    extract_unique_triplets_from_normalized_triplet_file, normalize_string,
    normalize_strings_triplets)
from src.utils import create_dir_if_not_exists, read_json, save_dict_to_json
from src.const import BENCHMARK_TABLES, TASK_NAME
from prompts.triplets_normalization import normalization_user_prompt, normalization_system_prompt, normalization_system_prompt_gpt_4_tubo
from pydantic import BaseModel, Field
import json

NORMALIZED_TRIPLET_PART = {"Task": TASK_NAME}

MODEL_NAME = "openai-gpt-oss-120b"
triplets_normalization_model_mapper = {"gpt-4-turbo": normalization_system_prompt_gpt_4_tubo,
                                       "openai-gpt-oss-120b": normalization_system_prompt}
from src.logger import logger

class NormalizationOutput(BaseModel):
    output_data_item: str = Field(description="A data item from the provided list or 'None' if the provided data item is not in the provided data items list. ") 

def combine_extracted_triplets_dir_into_file(extracted_triplets_dir: str) -> dict:
    combined_triplets_dict = {}

    if Path(extracted_triplets_dir).is_dir():
        for paper in Path(extracted_triplets_dir).iterdir():
            if paper.is_dir():
                papers_triplets = read_json(
                    Path(os.path.join(paper, "unique_triplets.json"))
                )
                combined_triplets_dict.update({paper.name: papers_triplets})
                # papers_triplets = []
                # for extracted_triplet_file_path in paper.iterdir():
                #     extracted_triplets = read_json(extracted_triplet_file_path)
                #     papers_triplets += extracted_triplets
                # combined_triplets_dict.update({paper.name: papers_triplets})

    return combined_triplets_dict


def create_labels_dict(true_dataset: dict) -> dict:
    """
    Create a dictionary of unique labels from the true dataset.

    Args:
        true_dataset: Dictionary containing ground truth TDM annotations per paper

    Returns:
        Dictionary with sets of unique Task, Dataset, and Metric labels
    """
    labels_dict = {"Task": set(), "Dataset": set(), "Metric": set()}

    for paper in true_dataset:
        for item in true_dataset[paper]["TDMs"]:
            labels_dict["Task"].add(item["Task"])
            labels_dict["Dataset"].add(item["Dataset"])
            labels_dict["Metric"].add(item["Metric"])

    return labels_dict


def main(
    path_to_extracted_triplets: str, true_dataset_path: str, output_dir_path: str, keys_to_normalize: set = {"Task", "Dataset", "Metric"},
) -> list[dict]:
    all_extracted_triplets_per_paper = combine_extracted_triplets_dir_into_file(
        path_to_extracted_triplets
    )  # read_json(Path(os.path.join(path_to_extracted_triplets, "unique_triplets.json")))
    true_dataset = read_json(Path(true_dataset_path))

    labels_dict = create_labels_dict(true_dataset)
    normalized_triplets = []

    already_processed_paper_names = [x.name for x in Path(output_dir_path).iterdir()]
    for paper_name in tqdm(all_extracted_triplets_per_paper):

        if paper_name in already_processed_paper_names:
            logger.info(f"The analyzed file was already processed: {paper_name}")
            continue

        print(f"Analyzed paper name: {paper_name}")
        output_paper_path = os.path.join(output_dir_path, paper_name)
        create_dir_if_not_exists(Path(output_paper_path))

        normalized_triplets_per_paper = []
        extracted_triplets_per_paper = all_extracted_triplets_per_paper[paper_name]

        model_system_prompt = triplets_normalization_model_mapper[MODEL_NAME]

        for extracted_triplet in extracted_triplets_per_paper:

            normalized_triplet = {k: NORMALIZED_TRIPLET_PART[k] for k, v in extracted_triplet.items() if k not in keys_to_normalize}
            for triplet_item in keys_to_normalize:
                try:
                    user_prompt = PromptTemplate.from_template(normalization_user_prompt).format(
                        data_item=extracted_triplet[triplet_item],
                        defined_list=labels_dict[triplet_item.capitalize()],
                    )
                    response = get_llm_model_response(prompt=user_prompt, model_name=MODEL_NAME, system_prompt=model_system_prompt,
                                                      pydantic_object_structured_output=NormalizationOutput)
                    if isinstance(response, NormalizationOutput):
                        response = response.output_data_item
                    elif isinstance(response, dict):
                        response = response["output_data_item"]

                    if response:
                        if response.lower() != "None".lower():
                            logger.info(f"Provided data item: {extracted_triplet[triplet_item]} output: {response}")
                            response = normalize_string(response)
                            normalized_triplet[triplet_item.capitalize()] = response

                        else:
                            logger.warning(
                                f"Model could not find the match for the {extracted_triplet[triplet_item]} within {labels_dict[triplet_item]}"
                                f" for paper {paper_name} and triplet: {extracted_triplet}"
                            )
                            logger.warning(response)


                except Exception as e:
                    logger.error(
                        f"Here is the exception: {e} for the {paper_name} and for {triplet_item} "
                        f"extracted_triplet: {extracted_triplet} and labels_dict: {labels_dict}"
                    )
                    normalized_triplet = extracted_triplet

            if len(normalized_triplet) == 3:
                normalized_triplets_per_paper.append(normalized_triplet)
                normalized_triplets.append(normalized_triplet)
            else:
                logger.error(f"The normalized triplet is broken: {normalized_triplet}, original triplet: {extracted_triplet}")

        save_dict_to_json(
            normalized_triplets_per_paper,
            Path(os.path.join(output_paper_path, paper_name + ".json")),
        )

    return normalized_triplets


def calculate_exact_match_on_extracted_triplets(gold_data, normalized_output):
    """

    Args:
        gold_data: A ground truth daa provided by the authors
        normalized_output: A normalized output provided by the authors

    Returns:

    """
    recall_scores = {}
    recall_scores_given_list = {}
    precision_scores = {}
    precision_scores_given_list = {}
    for paper in normalized_output:
        exact_matches = 0
        matches_within_list_of_extracted_values = 0
        # unique_output_tdms = [dict(t) for t in {tuple(d.items()) for d in output_tdms[paper]}]
        # for output_tdm in unique_output_tdms:
        for output_tdm in normalized_output[paper]['normalized_output']:
            found = False
            for gold_tdm in gold_data[paper]['TDMs']:
                try:
                    if (output_tdm['Task'] == gold_tdm['Task']) and (output_tdm['Dataset'] == gold_tdm['Dataset']) and (output_tdm['Metric'] == gold_tdm['Metric']):
                        exact_matches += 1
                        found = True
                    if found:
                        break
                except KeyError:
                    if (output_tdm['task'] == gold_tdm['Task']) and (output_tdm['dataset'] == gold_tdm['Dataset']) and (output_tdm['metric'] == gold_tdm['Metric']):
                        exact_matches += 1
                        found = True
                    if found:
                        break

        try:
            recall_scores[paper] = exact_matches / len(gold_data[paper]['TDMs'])
            recall_scores_given_list[paper] = matches_within_list_of_extracted_values / len(gold_data[paper]['TDMs'])
        except KeyError:
            recall_scores[paper] = 0.0
            recall_scores_given_list[paper] = 0.0
        if len(normalized_output[paper]['normalized_output']) != 0:
            precision_scores[paper] = exact_matches / len(normalized_output[paper]['normalized_output'])
            precision_scores_given_list[paper] = matches_within_list_of_extracted_values / len(normalized_output[paper]['normalized_output'])
        else:
            precision_scores[paper] = 0
            precision_scores_given_list[paper] = 0
    print(f"Overall recall score: {sum(recall_scores.values()) / len(recall_scores)}")
    print(f"Overall precision score: {sum(precision_scores.values()) / len(precision_scores)}")
    return recall_scores, precision_scores


def create_triplets_in_evaluation_form(triplets_normalization_output_dir: str) -> dict:
    triplets_in_correct_from = {}
    for paper in Path(triplets_normalization_output_dir).iterdir():
        paper_triplets_path = Path(os.path.join(str(paper), paper.name + ".json"))
        if paper_triplets_path.exists():
            papers_triplets = read_json(paper_triplets_path)
            triplets_in_correct_from[paper.name + ".pdf"] = {"normalized_output": papers_triplets}
        else:
            print(paper_triplets_path)

    return triplets_in_correct_from


def load_all_json_dicts(base_dir):
    all_dicts = []

    # Iterate through all subdirectories
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)

        if os.path.isdir(subdir_path):
            # Look for the JSON file in this directory
            for file_name in os.listdir(subdir_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(subdir_path, file_name)

                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            # Each JSON file is a list of dicts
                            if isinstance(data, list):
                                all_dicts.extend(data)
                            else:
                                print(f"Warning: {file_path} does not contain a list, skipping.")
                        except json.JSONDecodeError as e:
                            print(f"Error decoding {file_path}: {e}")

    return all_dicts


if __name__ == "__main__":
    normalization_output_dir = (
        "triplets_normalization/openai-gpt-oss-120b/chunk_focus_approach/custom_dataset/08_11_custom_dataset_dbpedia"
    )
    create_dir_if_not_exists(Path(normalization_output_dir))

    extracted_triplets_dir_path = (
        "triplets_extraction/chunk_focus_approach/openai-gpt-oss-120b/08_11_custom_dataset_dbpedia"
    )
    if not Path(extracted_triplets_dir_path).exists():
        raise FileNotFoundError(f"The provided path to extracted triplets: {extracted_triplets_dir_path} is broken!")

    true_dataset_path = "custom_dataset_papers/dbpedia/overall-result.json" # "leaderboard-generation/tdm_annotations.json"
    keys_to_normalize = {"Metric", "Dataset"}
    normalized_triplets = main(
        extracted_triplets_dir_path, true_dataset_path, normalization_output_dir, keys_to_normalize=keys_to_normalize
    )
    normalized_strings_triplets = normalize_strings_triplets(normalized_triplets)
    unique_triplets = extract_unique_triplets_from_normalized_triplet_file(
        normalized_strings_triplets
    )
    save_dict_to_json(
        unique_triplets,
        os.path.join(normalization_output_dir, "normalized_triplets.json"),
    )
    # Evaluation of normalized triplets
    current_triplets_evaluation_form = create_triplets_in_evaluation_form(normalization_output_dir)
    ground_truth_triplets = read_json(Path(true_dataset_path))
    recall_scores, precision_scores = calculate_exact_match_on_extracted_triplets(ground_truth_triplets,
                                                                                  current_triplets_evaluation_form)
