"""
Pipeline for TDMR Extraction with Author Approach
==================================================
This pipeline extends the base TDMR extraction pipeline with author approach methodology:
1. Prompt Extraction - Extract triplets from markdown files using LLM
2. Prompt Normalization - Normalize extracted triplets against reference dataset
3. TDMR Extraction (Author Approach) - Extract results using author model names
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from dotenv import load_dotenv

from src.tdmr_extraction.tdmr_extraction_with_author_approach import (
    main as extract_tdmr_with_author,
    define_all_unique_extracted_approaches_names,
)
from src.tdmr_extraction_utils.data_models import TdmrExtractionResponseWithModel
from prompts.tdmr_extracton_with_model_name import (
    TDMR_EXTRACTION_SYSTEM_PROMPT,
    TDMR_EXTRACTION_USER_PROMPT,
)
from src.tdmr_extraction_utils.author_model_extraction import (
    extract_author_model_prediction,
    combine_all_sections_based_json_into_one_file,
    extract_all_model_names_from_tables,
)
from src.parsers.docling_parsers import convert_pdf_into_md_using_docling
from src.tdmr_extraction_utils.utils import (
    chunk_markdown_file,
    create_one_result_file,
    create_one_result_file_for_evaluation_purpose,
)
from prompts.author_model_extraction import (
    EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_SYSTEM_PROMPT,
    EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT,
    EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT_WITH_TABLE_CONTEXT,
)
from src.utils import (
    create_dir_if_not_exists,
    read_json,
    save_dict_to_json,
    normalize_results_in_tdm_list,
    extract_tables_and_captions_from_pdf,
    save_str_as_markdown,
)
from src.logger import logger

# Reuse shared pipeline steps from updated_main
from updated_main import (
    extract_prompts_from_markdown,
    normalize_extracted_prompts,
    _is_step_complete,
)

load_dotenv()


# ============================================================================
# STEP 3 (ALTERNATIVE): TDMR EXTRACTION WITH AUTHOR APPROACH
# ============================================================================

def extract_tdmr_results_with_author_approach(
    normalized_triplets_dir: Path,
    papers_dir: Path,
    author_model_extraction_dir: Path,
    output_dir: Path,
    tables_output_dir: Path,
    model_name: str = "openai-gpt-oss-120b",
) -> Path:
    """
    Extract TDMR results using author approach methodology.

    This approach extracts author model names from papers and uses them
    when extracting TDMR results, providing additional context.

    Args:
        normalized_triplets_dir: Directory with normalized triplets
        papers_dir: Directory containing PDF papers
        author_model_extraction_dir: Directory with author model extraction results
        output_dir: Directory to save TDMR extraction results
        tables_output_dir: Directory containing pre-extracted tables (from Step 0b).
                          If provided, uses pre-extracted tables from this location.
                          If None, extracts tables on-the-fly to paper_output_dir.
        model_name: LLM model name to use

    Returns:
        Path to the output directory with TDMR results
    """
    logger.info("=" * 80)
    logger.info("STEP 3: TDMR EXTRACTION (WITH AUTHOR APPROACH)")
    logger.info("=" * 80)

    create_dir_if_not_exists(output_dir)

    # Get list of papers with normalized triplets (output from step 2)
    papers_with_normalized_triplets = [
        p for p in normalized_triplets_dir.iterdir() if p.is_dir()
    ]

    logger.info(f"Found {len(papers_with_normalized_triplets)} papers with normalized triplets")

    already_processed_files = [
        paper_path.name for paper_path in output_dir.iterdir()
    ] if output_dir.exists() else []

    for normalized_paper_path in tqdm(papers_with_normalized_triplets, desc="Extracting TDMR with Author Approach"):
        paper_name = normalized_paper_path.name
        logger.info(f"Processing paper: {paper_name}")

        if paper_name in already_processed_files:
            logger.info(f"Paper {paper_name} already processed, skipping...")
            continue

        # Create output directory for this paper
        paper_output_dir = output_dir / paper_name
        create_dir_if_not_exists(paper_output_dir)

        # Check if paper PDF exists
        paper_pdf_path = papers_dir / f"{paper_name}.pdf"
        if not paper_pdf_path.exists():
            logger.warning(f"No PDF found for {paper_name} at {paper_pdf_path}")
            continue

        # Check if author model extraction exists for this paper
        author_model_file = author_model_extraction_dir / paper_name / "author_model_approaches.json"
        if not author_model_file.exists():
            logger.info(f"No author model extraction found for {paper_name}, extracting now...")

            # Create output directory for author model extraction
            author_model_paper_output_dir = author_model_extraction_dir / paper_name
            create_dir_if_not_exists(author_model_paper_output_dir)

            try:
                # Convert PDF to markdown using docling
                logger.info(f"Converting PDF to markdown for {paper_name}...")
                md_file_path = convert_pdf_into_md_using_docling(paper_pdf_path)

                # Extract model names from pre-extracted tables for this paper
                # Set use_ollama=False to fall back to the API-based model
                paper_tables_dir = tables_output_dir / paper_name if tables_output_dir else None
                table_model_names = extract_all_model_names_from_tables(
                    model_name,
                    use_ollama=False,
                    tables_dir=paper_tables_dir,
                    md_file_path=md_file_path,
                )

                # Choose the appropriate user prompt based on whether table names were found
                if table_model_names:
                    active_user_prompt = EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT_WITH_TABLE_CONTEXT
                else:
                    active_user_prompt = EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT

                # Chunk the markdown file
                logger.info(f"Chunking markdown file for {paper_name}...")
                papers_section_text = chunk_markdown_file(md_file_path, 4000)

                if not papers_section_text:
                    logger.warning(f"Could not chunk markdown file for {paper_name}")
                    continue

                # Process each section/chunk
                for section_key, section_content in papers_section_text.items():
                    if section_content:
                        # Create temporary markdown file in current working directory
                        # (matching the original behavior in author_model_extraction.py)
                        temp_md_filename = f"{section_key}_len{len(section_content)}.md"
                        save_str_as_markdown(temp_md_filename, section_content)

                        # Extract author model prediction for this section
                        extract_author_model_prediction(
                            markdown_file_path=temp_md_filename,
                            output_dir=str(author_model_paper_output_dir),
                            model_name=model_name,
                            system_prompt=EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_SYSTEM_PROMPT,
                            user_prompt=active_user_prompt,
                            table_model_names=table_model_names if table_model_names else None,
                        )

                        # Remove temporary markdown file from current working directory
                        temp_md_path = Path(temp_md_filename)
                        if temp_md_path.exists():
                            temp_md_path.unlink()
                    else:
                        logger.warning(f"Empty content for section {section_key} in {paper_name}")

                # Combine all section results into one file
                combine_all_sections_based_json_into_one_file(
                    output_dir_path=str(author_model_paper_output_dir)
                )

                logger.info(f"Author model extraction completed for {paper_name}")

                # Clean up the generated markdown file
                md_file_to_remove = Path(md_file_path)
                if md_file_to_remove.exists():
                    md_file_to_remove.unlink()

            except Exception as e:
                logger.error(f"Error extracting author model for {paper_name}: {str(e)}")
                continue

            # Verify the author model file was created
            if not author_model_file.exists():
                logger.warning(f"Author model extraction failed for {paper_name}, file not created")
                continue

        try:
            # Load author model approaches
            list_of_all_extracted_approaches_per_section = read_json(author_model_file)
            ## TODO what if no model was extracted?  INSTRUCT THE MODEL TO CHOOSE ONLY FROM EXTRAED TABELS AND ALSO THTA IT HAS TO THE MODEL OR SOMETHING NOT EVALAUTION SUITE
            # Handle different formats of author model extraction
            list_of_all_extracted_approaches_per_section = (
                [list_of_all_extracted_approaches_per_section[-1]]
                if "most_common_model_name" in list_of_all_extracted_approaches_per_section[-1]
                else list_of_all_extracted_approaches_per_section
            )

            # Extract unique author approaches
            all_extracted_authors_approach = (
                define_all_unique_extracted_approaches_names(
                    list_of_all_extracted_approaches_per_section
                )
                if len(list_of_all_extracted_approaches_per_section) > 1
                else [
                    list_of_all_extracted_approaches_per_section[0]["most_common_model_name"]
                ]
            )

            if not all_extracted_authors_approach:
                logger.warning(f"No author approaches found for {paper_name}")
                continue

            logger.info(f"Found {len(all_extracted_authors_approach)} unique author approaches for {paper_name}")

            # Use pre-extracted tables from tables_output_dir (created in Step 0b)
            if tables_output_dir:
                tables_paper_dir = tables_output_dir / paper_name
                tables_result_file = tables_paper_dir / "result_dict.json"
                if tables_result_file.exists():
                    extracted_tables = str(tables_result_file)
                    logger.info(f"Using pre-extracted tables from {tables_result_file}")
                else:
                    logger.warning(f"No pre-extracted tables found for {paper_name}, extracting now...")
                    extracted_tables = extract_tables_and_captions_from_pdf(str(paper_pdf_path), str(tables_output_dir))
            else:
                # Fallback: extract tables to paper output directory
                logger.info(f"No tables_output_dir provided, extracting tables for {paper_name}...")
                extracted_tables = extract_tables_and_captions_from_pdf(str(paper_pdf_path), str(paper_output_dir / "tables"))

            # Run TDMR extraction with author approach
            extract_tdmr_with_author(
                extracted_triplet_path_dir=str(normalized_paper_path),
                all_extracted_author_approach=all_extracted_authors_approach,
                extracted_tables_summary_file_path=extracted_tables,
                tdmr_extraction_output_dir_path=paper_output_dir,
                user_prompt=TDMR_EXTRACTION_USER_PROMPT,
                system_prompt=TDMR_EXTRACTION_SYSTEM_PROMPT,
                pydantic_object=TdmrExtractionResponseWithModel,
                model_name=model_name,
            )

        except Exception as e:
            logger.error(f"Error processing {paper_name}: {str(e)}")
            continue

    logger.info(f"Step 3 (Author Approach) completed. Output saved to: {output_dir}")
    return output_dir


# ============================================================================
# RESULT NORMALIZATION
# ============================================================================

def normalize_pipeline_results(output_dir: Path) -> None:
    """
    Normalize all Result values in the pipeline output to 0-1 decimal scale.

    Args:
        output_dir: Directory containing the pipeline output files
    """
    eval_results_file = output_dir / "processed_tdmr_extraction_test_papers_evaluation_valid_results.json"

    if not eval_results_file.exists():
        logger.warning(f"Evaluation results file not found: {eval_results_file}")
        return

    logger.info("Normalizing pipeline results to 0-1 decimal scale...")
    results = read_json(eval_results_file)

    for paper_name in results.keys():
        if "normalized_output" in results[paper_name]:
            results[paper_name]["normalized_output"] = normalize_results_in_tdm_list(
                results[paper_name]["normalized_output"]
            )

    save_dict_to_json(results, str(eval_results_file))
    logger.info(f"Normalized results saved to: {eval_results_file}")


# ============================================================================
# MAIN PIPELINE WITH AUTHOR EXTENSION
# ============================================================================

def run_complete_pipeline_with_authors_extension(
    author_model_extraction_dir: str,
    true_dataset_path: str,
    base_output_dir: str,
    pdf_files_dir: Optional[str] = None,
    markdown_files_dir: Optional[str] = None,
    papers_dir: Optional[str] = None,
    model_name: str = "openai-gpt-oss-120b",
    keys_to_normalize: Optional[set] = None,
    chunk_size: int = 5000,
    resume_from_dir: Optional[str] = None,
):
    """
    Run the complete TDMR extraction pipeline with author approach extension.

    This pipeline combines:
    1. Triplet extraction from markdown files
    2. Triplet normalization against reference dataset
    3. TDMR extraction using author approach methodology

    Args:
        author_model_extraction_dir: Directory with author model extraction results
        true_dataset_path: Path to ground truth dataset JSON
        base_output_dir: Base directory for all outputs
        pdf_files_dir: Directory containing PDF files (optional, primary input)
        markdown_files_dir: Directory containing markdown files (optional, will be created from PDFs if not provided)
        papers_dir: Directory containing PDF papers (optional, defaults to pdf_files_dir)
        model_name: LLM model name to use
        keys_to_normalize: Set of keys to normalize (default: {"Metric", "Dataset"})
        chunk_size: Size of text chunks for extraction (default: 5000)
        resume_from_dir: Optional directory name to resume from (e.g., "26_12_2025").
                        If provided, the pipeline will resume from this directory
    """
    # Determine output directory
    if resume_from_dir:
        # Use the specified directory for resuming
        base_output_path = Path(base_output_dir) / model_name / resume_from_dir

        if not base_output_path.exists():
            logger.error(f"Resume directory does not exist: {base_output_path}")
            raise ValueError(f"Cannot resume from non-existent directory: {base_output_path}")

        logger.info("=" * 80)
        logger.info(f"RESUMING FROM DIRECTORY: {base_output_path}")
        logger.info("=" * 80)
    else:
        # Create new timestamp-based directory
        timestamp = datetime.now().strftime("%d_%m_%Y")
        base_output_path = Path(base_output_dir) / model_name / timestamp

        # Check if this day's run already exists
        if base_output_path.exists():
            logger.info("=" * 80)
            logger.info(f"Output directory {base_output_path} already exists")
            logger.info("Continuing with existing run - papers already processed will be skipped")
            logger.info("=" * 80)
        else:
            create_dir_if_not_exists(base_output_path)

    # Step 0: Handle PDF conversion if needed
    if pdf_files_dir:
        pdf_files_path = Path(pdf_files_dir)

        # Default papers_dir to pdf_files_dir if not specified
        if not papers_dir:
            papers_dir = pdf_files_dir

        # Convert PDFs to markdown if markdown_files_dir not specified
        if not markdown_files_dir:
            logger.info("=" * 80)
            logger.info("STEP 0: Converting PDFs to Markdown")
            logger.info("=" * 80)

            # Create markdown directory next to PDFs
            markdown_output_dir = pdf_files_path / "markdowns"
            create_dir_if_not_exists(markdown_output_dir)

            # Convert PDFs to markdown
            pdf_files = list(pdf_files_path.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files to convert")

            for pdf_file in tqdm(pdf_files, desc="Converting PDFs to markdown"):
                md_file_path = markdown_output_dir / f"{pdf_file.stem}.md"

                # Skip if already exists
                if md_file_path.exists():
                    logger.info(f"Skipping {pdf_file.name} - markdown already exists")
                    continue

                try:
                    logger.info(f"Converting {pdf_file.name} to markdown...")
                    temp_md_path = convert_pdf_into_md_using_docling(Path(pdf_file))

                    # Move to output directory
                    import shutil
                    shutil.move(temp_md_path, md_file_path)
                    logger.info(f"Successfully converted {pdf_file.name}")
                except Exception as e:
                    logger.error(f"Error converting {pdf_file.name}: {str(e)}")
                    continue

            markdown_files_dir = str(markdown_output_dir)
            logger.info(f"Markdown files created in: {markdown_files_dir}")

        # Extract tables from PDFs (Step 0b - matching run_complete_pipeline)
        tables_output_dir = pdf_files_path / "tables"
        if not tables_output_dir.exists() or not list(tables_output_dir.glob("*/result_dict.json")):
            logger.info("=" * 80)
            logger.info("STEP 0b: Extracting Tables from PDFs")
            logger.info("=" * 80)

            create_dir_if_not_exists(tables_output_dir)

            pdf_files = list(pdf_files_path.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files to extract tables from")

            for pdf_file in tqdm(pdf_files, desc="Extracting tables from PDFs"):
                paper_tables_dir = tables_output_dir / pdf_file.stem
                result_dict_path = paper_tables_dir / "result_dict.json"

                # Skip if already exists
                if result_dict_path.exists():
                    logger.info(f"Skipping {pdf_file.name} - tables already extracted")
                    continue

                try:
                    logger.info(f"Extracting tables from {pdf_file.name}...")
                    extract_tables_and_captions_from_pdf(
                        pdf_path=str(pdf_file),
                        output_dir=str(tables_output_dir)
                    )
                    logger.info(f"Successfully extracted tables from {pdf_file.name}")
                except Exception as e:
                    logger.error(f"Error extracting tables from {pdf_file.name}: {str(e)}")
                    continue

            logger.info(f"Tables extracted to: {tables_output_dir}")
        else:
            logger.info(f"Tables directory already exists at: {tables_output_dir}")
    else:
        tables_output_dir = None

    # Validate that we have the required directories
    if not markdown_files_dir:
        if not papers_dir:
            raise ValueError("Either papers_dir or pdf_files_dir must be provided")

    extraction_output_dir = base_output_path / "01_triplets_extraction"
    normalization_output_dir = base_output_path / "02_triplets_normalization"
    tdmr_output_dir = base_output_path / "03_tdmr_extraction_with_author_approach"

    # Detect which steps have been completed
    markdown_files_path = Path(markdown_files_dir)
    step1_complete = _is_step_complete(extraction_output_dir, markdown_files_path)
    step2_complete = _is_step_complete(normalization_output_dir, markdown_files_path)
    step3_complete = _is_step_complete(tdmr_output_dir, markdown_files_path)

    logger.info("=" * 80)
    logger.info("STARTING COMPLETE TDMR EXTRACTION PIPELINE (WITH AUTHOR APPROACH)")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Base output directory: {base_output_path}")

    if resume_from_dir:
        logger.info(f"Resume mode: Analyzing existing directory")
        logger.info(f"  Step 1 (Triplet Extraction): {'COMPLETE' if step1_complete else 'INCOMPLETE'}")
        logger.info(f"  Step 2 (Normalization): {'COMPLETE' if step2_complete else 'INCOMPLETE'}")
        logger.info(f"  Step 3 (TDMR Extraction with Author Approach): {'COMPLETE' if step3_complete else 'INCOMPLETE'}")

    try:
        # Step 1: Extract triplets from markdown files
        if step1_complete and resume_from_dir:
            logger.info("=" * 80)
            logger.info("STEP 1: SKIPPING (Already completed)")
            logger.info("=" * 80)
        else:
            extract_prompts_from_markdown(
                markdown_files_dir=Path(markdown_files_dir),
                output_dir=extraction_output_dir,
                model_name=model_name,
                chunk_size=chunk_size,
            )

        # Step 2: Normalize extracted triplets
        if step2_complete and resume_from_dir:
            logger.info("=" * 80)
            logger.info("STEP 2: SKIPPING (Already completed)")
            logger.info("=" * 80)
        else:
            normalize_extracted_prompts(
                extracted_triplets_dir=extraction_output_dir,
                true_dataset_path=Path(true_dataset_path),
                output_dir=normalization_output_dir,
                model_name=model_name,
                keys_to_normalize=keys_to_normalize,
            )

        # Step 3: Extract TDMR results with author approach
        if step3_complete and resume_from_dir:
            logger.info("=" * 80)
            logger.info("STEP 3: SKIPPING (Already completed)")
            logger.info("=" * 80)
        else:
            extract_tdmr_results_with_author_approach(
                normalized_triplets_dir=normalization_output_dir,
                papers_dir=Path(papers_dir),
                author_model_extraction_dir=Path(author_model_extraction_dir),
                output_dir=tdmr_output_dir,
                model_name=model_name,
                tables_output_dir=tables_output_dir,
            )

            # Create consolidated result files
            logger.info("Creating consolidated result files...")
            create_one_result_file(tdmr_output_dir)
            create_one_result_file_for_evaluation_purpose(tdmr_output_dir)
            logger.info("Consolidated result files created successfully")

        # Step 4: Normalize results to 0-1 decimal scale
        normalize_pipeline_results(tdmr_output_dir)

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"All outputs saved to: {base_output_path}")
        logger.info(f"  - Triplets extraction: {extraction_output_dir}")
        logger.info(f"  - Triplets normalization: {normalization_output_dir}")
        logger.info(f"  - TDMR extraction (with author approach): {tdmr_output_dir}")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    CONFIG = {
        "pdf_files_dir": "custom_dataset_papers_refined/wikidata/all_papers_wikidata",
        "author_model_extraction_dir": "author_model_extraction",
        "markdown_files_dir": "",
        "true_dataset_path": "custom_dataset_papers_refined/wikidata/all_papers_wikidata/all_dbpedia.json",
        "base_output_dir": "pipeline_results_with_author_approach",
        "model_name": "openai-gpt-oss-120b",
        "keys_to_normalize": {"Metric", "Dataset"},
        "chunk_size": 5000,
        "resume_from_dir": "15_03_2026_all_wikidata",
    }

    run_complete_pipeline_with_authors_extension(**CONFIG)