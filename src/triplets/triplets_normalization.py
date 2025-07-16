import os
from pathlib import Path

from langchain.prompts import PromptTemplate
from tqdm import tqdm

from src.openai_client import get_openai_model_response
from src.triplets.triplets_unification import (
    extract_unique_triplets_from_normalized_triplet_file, normalize_string,
    normalize_strings_triplets)
from src.utils import create_dir_if_not_exists, read_json, save_dict_to_json

MODEL_NAME = "gpt-4o"

from src.logger import logger

normalization_prompt = """
You will be given with a data item and a list of similar items. Your task is to decide, whether this provided data item is a new item
or whether it's already defined in different name in the provided list. If it's a new item (it does not exist in the provided list), please return None. If the 
provided list already contains this data item in different name, please provide the defined version.
Do not provide any further explanations!
Please take a note that the new item doesn't have to be exactly 1:1 with the defined list, it's more about the meaning.
Additionally, please make sure that when the data item already exists you MUST provide the value from the defined list, not the original data item.

Here is the data item (input):
{data_item}

Here is the defined list:
{defined_list}
"""


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


def main(
    path_to_extracted_triplets: str, true_dataset_path: str, output_dir_path: str
) -> list[dict]:
    all_extracted_triplets_per_paper = combine_extracted_triplets_dir_into_file(
        path_to_extracted_triplets
    )  # read_json(Path(os.path.join(path_to_extracted_triplets, "unique_triplets.json")))
    true_dataset = read_json(Path(true_dataset_path))

    labels_dict = {"Task": set(), "Dataset": set(), "Metric": set()}
    normalized_triplets = []

    for paper in true_dataset:
        for item in true_dataset[paper]["TDMs"]:
            labels_dict["Task"].add(item["Task"])
            labels_dict["Dataset"].add(item["Dataset"])
            labels_dict["Metric"].add(item["Metric"])

    already_processed_paper_names = [x.name for x in Path(output_dir_path).iterdir()]
    for paper_name in tqdm(all_extracted_triplets_per_paper):

        if paper_name in already_processed_paper_names:
            logger.warning(f"The analyzed file was already processed: {paper_name}")
            continue

        print(f"Analyzed paper name: {paper_name}")
        output_paper_path = os.path.join(output_dir_path, paper_name)
        create_dir_if_not_exists(Path(output_paper_path))

        normalized_triplets_per_paper = []
        extracted_triplets_per_paper = all_extracted_triplets_per_paper[paper_name]

        for extracted_triplet in extracted_triplets_per_paper:

            normalized_triplet = {}
            for triplet_item in extracted_triplet:

                try:

                    prompt = PromptTemplate.from_template(normalization_prompt).format(
                        data_item=extracted_triplet[triplet_item],
                        defined_list=labels_dict[triplet_item],
                    )
                    response = get_openai_model_response(prompt, model_name=MODEL_NAME)

                    if response.lower() != "None".lower():
                        print(response)
                        response = normalize_string(response)
                        normalized_triplet[triplet_item] = response

                    else:
                        logger.warning(
                            f"Model could not find the match for the {extracted_triplet[triplet_item]} within {labels_dict[triplet_item]}"
                            f"for paper {paper_name} and triplet: {extracted_triplet}"
                        )
                        logger.warning(response)
                        normalized_triplet[triplet_item] = extracted_triplet[
                            triplet_item
                        ]

                except Exception as e:
                    logger.error(
                        f"Here is the exception: {e} for the {paper_name} and for {triplet_item}"
                        f"extracted_triplet: {extracted_triplet} and labels_dict: {labels_dict}"
                    )
                    print(extracted_triplet)
                    print(labels_dict)
                    normalized_triplet = extracted_triplet

            normalized_triplets_per_paper.append(normalized_triplet)
            normalized_triplets.append(normalized_triplet)

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
                if (output_tdm['Task'] == gold_tdm['Task']) and (output_tdm['Dataset'] == gold_tdm['Dataset']) and (output_tdm['Metric'] == gold_tdm['Metric']):
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


if __name__ == "__main__":
    normalization_output_dir = (
        "triplets_normalization/from_entire_document_refined_prompt_gpt-4-turbo_more_context_less_strict"
    )
    current_triplets_evaluation_form = create_triplets_in_evaluation_form(normalization_output_dir)
    ground_truth_triplets = read_json(Path("leaderboard-generation/tdm_annotations.json"))
    recall_scores, precision_scores = calculate_exact_match_on_extracted_triplets(ground_truth_triplets, current_triplets_evaluation_form)
    create_dir_if_not_exists(Path(normalization_output_dir))
    extracted_triplets_dir_path = (
        "triplets_extraction/from_entire_document_refined_prompt_gpt-4-turbo_more_context_less_strict_prompt"
    )
    create_dir_if_not_exists(Path(extracted_triplets_dir_path))

    true_dataset_path = "leaderboard-generation/tdm_annotations.json"
    normalized_triplets = main(
        extracted_triplets_dir_path, true_dataset_path, normalization_output_dir
    )
    normalized_strings_triplets = normalize_strings_triplets(normalized_triplets)
    unique_triplets = extract_unique_triplets_from_normalized_triplet_file(
        normalized_strings_triplets
    )
    save_dict_to_json(
        unique_triplets,
        os.path.join(normalization_output_dir, "normalized_triplets.json"),
    )
