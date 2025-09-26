import os
from pathlib import Path

from src.utils import read_json, save_dict_to_json


def extract_unique_triplets_from_normalized_triplet_file(data: list[dict]):
    return [dict(t) for t in {tuple(sorted(d.items())) for d in data}]


def deduplicate_outputs(normalized_output: list[dict]):
    seen = set()
    deduped = []
    for entry in normalized_output:
        key = (entry.get("Task"), entry.get("Dataset"), entry.get("Metric"))
        if key not in seen:
            seen.add(key)
            deduped.append(entry)
    return deduped


def extract_unique_triplets_from_evaluation_purpose_file(read_data: dict):
    for pdf_file, pdf_data in read_data.items():
        if "normalized_output" in pdf_data:
            pdf_data["normalized_output"] = deduplicate_outputs(
                pdf_data["normalized_output"]
            )
    return read_data


def normalize_string(s):
    # Strip quotes and whitespace from string
    return s.strip().strip("'").strip('"')


def normalize_strings_triplets(triplets: list[dict]):
    for triplet in triplets:
        for k, v in triplet.items():
            try:
                triplet[k] = normalize_string(v)
            except:
                continue
    return triplets


def normalize_extracted_triplet_dir(extracted_triplet_dir: str):
    for paper_dir in Path(extracted_triplet_dir).iterdir():
        if paper_dir.is_dir():
            extracted_triplet = read_json(
                Path(os.path.join(paper_dir, f"{paper_dir.name}.json"))
            )
            normalized_strings_triplets = normalize_strings_triplets(extracted_triplet)
            unique_triplets = extract_unique_triplets_from_normalized_triplet_file(
                normalized_strings_triplets
            )
            save_dict_to_json(
                unique_triplets, Path(os.path.join(paper_dir, f"{paper_dir.name}.json"))
            )


if __name__ == "__main__":
    triplets_dir = "triplets_normalization_all_leaderboards_papers_gpt-4-turbo"
    normalize_extracted_triplet_dir(triplets_dir)
    # evaluation_purpose_file_path = "tdmr_extraction/gpt-4o/with_picking_optimal_table/processed_tdmr_extraction_test_papers_evaluation.json"
    # normalized_triplets = read_json(Path(evaluation_purpose_file_path))
    # unique_data = extract_unique_triplets_from_evaluation_purpose_file(normalized_triplets)
    # print(unique_data)
    # normalized_triplets_file_path = "triplets_normalization_all_leaderboards_papers_gpt-4-turbo/normalized_triplets.json"
    # normalized_triplets = read_json(Path(normalized_triplets_file_path))
    # unique_data = [dict(t) for t in {tuple(sorted(d.items())) for d in normalized_triplets}]
    # print(unique_data)
