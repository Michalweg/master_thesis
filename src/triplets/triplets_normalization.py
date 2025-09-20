import os
from pathlib import Path

from langchain.prompts import PromptTemplate
from tqdm import tqdm

from src.openai_client import get_llm_model_response
from src.triplets.triplets_unification import (
    extract_unique_triplets_from_normalized_triplet_file, normalize_string,
    normalize_strings_triplets)
from src.utils import create_dir_if_not_exists, read_json, save_dict_to_json
from src.const import BENCHMARK_TABLES
from prompts.triplets_normalization import normalization_user_prompt, normalization_system_prompt, normalization_system_prompt_gpt_4_tubo
from pydantic import BaseModel, Field

MODEL_NAME = "gpt-4-turbo"
triplets_normalization_model_mapper = {"gpt-4-turbo": normalization_system_prompt_gpt_4_tubo,
                                       "openai-gpt-oss-120b": normalization_user_prompt}
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


        # if paper_name not in BENCHMARK_TABLES:
        #     continue


        if paper_name in already_processed_paper_names:
            logger.warning(f"The analyzed file was already processed: {paper_name}")
            continue

        print(f"Analyzed paper name: {paper_name}")
        output_paper_path = os.path.join(output_dir_path, paper_name)
        create_dir_if_not_exists(Path(output_paper_path))

        normalized_triplets_per_paper = []
        extracted_triplets_per_paper = all_extracted_triplets_per_paper[paper_name]

        model_system_prompt = triplets_normalization_model_mapper[MODEL_NAME]

        for extracted_triplet in extracted_triplets_per_paper:

            normalized_triplet = {}
            for triplet_item in extracted_triplet:
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
                                f"for paper {paper_name} and triplet: {extracted_triplet}"
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


if __name__ == "__main__":
    normalization_output_dir = (
        f"triplets_normalization/{MODEL_NAME}/from_chunk_approach_refined_prompt_12_08"
    )
    create_dir_if_not_exists(Path(normalization_output_dir))

    extracted_triplets_dir_path = (
        f"triplets_extraction/chunk_focus_approach/gpt-4-turbo/12_08_update"
    )
    if not Path(extracted_triplets_dir_path).exists():
        raise FileNotFoundError(f"The provided path to extracted triplets: {extracted_triplets_dir_path} is broken!")

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
    # Evaluation of normalized triplets
    current_triplets_evaluation_form = create_triplets_in_evaluation_form(normalization_output_dir)
    ground_truth_triplets = read_json(Path("leaderboard-generation/tdm_annotations.json"))
    recall_scores, precision_scores = calculate_exact_match_on_extracted_triplets(ground_truth_triplets,
                                                                                  current_triplets_evaluation_form)
