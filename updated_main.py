"""
Combined Pipeline for TDMR Extraction
======================================
This pipeline combines three main steps:
1. Prompt Extraction - Extract triplets from markdown files using LLM
2. Prompt Normalization - Normalize extracted triplets against reference dataset
3. TDMR Extraction - Extract results from tables using normalized triplets
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from dotenv import load_dotenv

# Step 1: Prompt Extraction imports
from src.openai_client import (
    get_llm_model_response,
    ExtractedTriplets,
)
from prompts.triplets_extraction import (
    openai_gpt_oss_120b_system_prompt,
    openai_gpt_oss_120b_user_prompt,
)

# Step 2: Prompt Normalization imports
from src.triplets.triplets_normalization import main as normalize_triplets
from src.triplets.triplets_unification import (
    extract_unique_triplets_from_normalized_triplet_file,
    normalize_strings_triplets,
)

# Step 3: TDMR Extraction imports
from src.tdmr_extraction.tdmr_extraction_with_table_and_value_selection import (
    main as extract_tdmr,
)
from src.tdmr_extraction_utils.utils import (
    create_one_result_file,
    create_one_result_file_for_evaluation_purpose,
)

# Step 3 (Alternative): TDMR Extraction with Author Approach imports
from src.tdmr_extraction.tdmr_extraction_with_author_approach import (
    main as extract_tdmr_with_author,
    define_all_unique_extracted_approaches_names,
)
from src.tdmr_extraction_utils.data_models import TdmrExtractionResponseWithModel
from prompts.tdmr_extracton_with_model_name import (
    TDMR_EXTRACTION_SYSTEM_PROMPT,
    TDMR_EXTRACTION_USER_PROMPT,
)

# Author model extraction imports
from src.tdmr_extraction_utils.author_model_extraction import (
    extract_author_model_prediction,
    combine_all_sections_based_json_into_one_file,
)
from src.parsers.docling_parsers import convert_pdf_into_md_using_docling
from src.tdmr_extraction_utils.utils import chunk_markdown_file
from prompts.author_model_extraction import (
    EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_SYSTEM_PROMPT,
    EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT,
)

# Common utilities
from src.utils import (
    create_dir_if_not_exists,
    read_json,
    read_markdown_file_content,
    save_dict_to_json,
    convert_pdfs_to_markdown_in_directory,
    extract_tables_and_captions_from_pdf,
    save_str_as_markdown,
)
from src.logger import logger

load_dotenv()


# ============================================================================
# STEP 1: PROMPT EXTRACTION
# ============================================================================

def extract_prompts_from_markdown(
    markdown_files_dir: Path,
    output_dir: Path,
    model_name: str = "openai-gpt-oss-120b",
    chunk_size: int = 5000,
) -> Path:
    """
    Extract triplets from markdown files using chunk-based approach.

    Args:
        markdown_files_dir: Directory containing markdown files
        output_dir: Directory to save extracted triplets
        model_name: LLM model name to use
        chunk_size: Size of text chunks to process

    Returns:
        Path to the output directory with extracted triplets
    """
    logger.info("=" * 80)
    logger.info("STEP 1: PROMPT EXTRACTION")
    logger.info("=" * 80)

    create_dir_if_not_exists(output_dir)

    # Iterate through all markdown files/papers
    for paper_path in tqdm(list(markdown_files_dir.iterdir()), desc="Extracting triplets"):
        # Handle both direct .md files and directories containing .md files
        if paper_path.is_dir():
            paper_name = paper_path.name
            # Find markdown file in subdirectory
            markdown_file_candidates = list(paper_path.glob("*.md"))
            if not markdown_file_candidates:
                logger.warning(f"No markdown file found in directory {paper_name}")
                continue
            markdown_file_path = markdown_file_candidates[0]
        elif paper_path.suffix == ".md":
            # Direct markdown file
            paper_name = paper_path.stem
            markdown_file_path = paper_path
        else:
            # Skip non-markdown files and non-directories
            continue

        logger.info(f"Processing paper: {paper_name}")

        # Create output directory for this paper
        paper_output_dir = output_dir / paper_name
        create_dir_if_not_exists(paper_output_dir)

        # Check if already processed
        unique_triplets_path = paper_output_dir / "unique_triplets.json"
        if unique_triplets_path.exists():
            logger.info(f"Paper {paper_name} already processed, skipping...")
            continue

        # Read markdown content
        file_content = read_markdown_file_content(markdown_file_path)
        if not file_content:
            logger.warning(f"Empty content in {markdown_file_path}")
            continue

        # Process in chunks
        valid_jsons_list = []
        chunks = [
            file_content[i : i + chunk_size]
            for i in range(0, len(file_content), chunk_size)
        ]

        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)} for {paper_name}")

            try:
                user_prompt = openai_gpt_oss_120b_user_prompt.format(
                    research_paper=chunk
                )

                response = get_llm_model_response(
                    prompt=user_prompt,
                    pydantic_object_structured_output=ExtractedTriplets,
                    system_prompt=openai_gpt_oss_120b_system_prompt,
                    model_name=model_name,
                )

                if isinstance(response, ExtractedTriplets):
                    for triplet in response.extracted_triplets:
                        valid_jsons_list.append(triplet.model_dump())
                elif isinstance(response, dict) and "extracted_triplets" in response:
                    valid_jsons_list.extend(response["extracted_triplets"])

            except Exception as e:
                logger.error(f"Error processing chunk {idx} for {paper_name}: {str(e)}")
                continue

        # Remove duplicates and save
        unique_triplets = list({
            tuple(sorted(d.items())) for d in valid_jsons_list
        })
        unique_triplets = [dict(t) for t in unique_triplets]

        save_dict_to_json(unique_triplets, unique_triplets_path)
        logger.info(f"Extracted {len(unique_triplets)} unique triplets for {paper_name}")

    # Unify triplets for each paper to remove any remaining duplicates
    logger.info("Unifying extracted triplets to remove duplicates...")
    total_before = 0
    total_after = 0

    for paper_dir in output_dir.iterdir():
        if paper_dir.is_dir():
            # Find the unique_triplets.json file for this paper
            triplet_file = paper_dir / "unique_triplets.json"
            if triplet_file.exists():
                triplets = read_json(triplet_file)
                total_before += len(triplets)

                # Remove duplicates by converting to tuples and back
                unique_triplets_set = {
                    tuple(sorted(d.items())) for d in triplets
                }
                unique_triplets_list = [dict(t) for t in unique_triplets_set]
                total_after += len(unique_triplets_list)

                # Save unified triplets back to the file
                save_dict_to_json(unique_triplets_list, triplet_file)

                if len(triplets) != len(unique_triplets_list):
                    logger.info(f"Paper {paper_dir.name}: Removed {len(triplets) - len(unique_triplets_list)} duplicates ({len(triplets)} -> {len(unique_triplets_list)})")

    logger.info(f"Unification complete: {total_before} -> {total_after} triplets (removed {total_before - total_after} duplicates)")

    logger.info(f"Step 1 completed. Output saved to: {output_dir}")
    return output_dir


# ============================================================================
# STEP 2: PROMPT NORMALIZATION
# ============================================================================

def normalize_extracted_prompts(
    extracted_triplets_dir: Path,
    true_dataset_path: Path,
    output_dir: Path,
    model_name: str = "openai-gpt-oss-120b",
    keys_to_normalize: Optional[set] = None,
) -> Path:
    """
    Normalize extracted triplets against reference dataset.

    Args:
        extracted_triplets_dir: Directory with extracted triplets
        true_dataset_path: Path to ground truth dataset JSON
        output_dir: Directory to save normalized triplets
        model_name: LLM model name to use
        keys_to_normalize: Set of keys to normalize (e.g., {"Metric", "Dataset"})

    Returns:
        Path to the output directory with normalized triplets
    """
    logger.info("=" * 80)
    logger.info("STEP 2: PROMPT NORMALIZATION")
    logger.info("=" * 80)

    create_dir_if_not_exists(output_dir)

    if keys_to_normalize is None:
        keys_to_normalize = {"Metric", "Dataset"}

    # Check which papers have already been normalized
    papers_to_process = []
    already_processed_papers = []

    for paper_dir in extracted_triplets_dir.iterdir():
        if paper_dir.is_dir():
            paper_name = paper_dir.name
            # Check if this paper already has normalized output
            normalized_paper_dir = output_dir / paper_name
            if normalized_paper_dir.exists() and list(normalized_paper_dir.glob("*.json")):
                already_processed_papers.append(paper_name)
                logger.info(f"Paper {paper_name} already normalized, skipping...")
            else:
                papers_to_process.append(paper_name)

    logger.info(f"Papers already normalized: {len(already_processed_papers)}")
    logger.info(f"Papers to process: {len(papers_to_process)}")

    # Only run normalization if there are papers to process
    if papers_to_process:
        # Run normalization
        normalized_triplets = normalize_triplets(
            path_to_extracted_triplets=str(extracted_triplets_dir),
            true_dataset_path=str(true_dataset_path),
            output_dir_path=str(output_dir),
            keys_to_normalize=keys_to_normalize,
        )
    else:
        logger.info("All papers already normalized, skipping normalization step")
        # Load existing normalized triplets from all papers
        normalized_triplets = []
        for paper_dir in output_dir.iterdir():
            if paper_dir.is_dir():
                json_files = list(paper_dir.glob("*.json"))
                if json_files:
                    paper_triplets = read_json(json_files[0])
                    normalized_triplets.extend(paper_triplets)

    # Unify triplets for each paper to remove duplicates (only for papers that were just processed)
    if papers_to_process:
        logger.info("Unifying triplets to remove duplicates...")
        total_before = 0
        total_after = 0

        for paper_name in papers_to_process:
            paper_dir = output_dir / paper_name
            if paper_dir.is_dir():
                # Find the JSON file for this paper
                json_files = list(paper_dir.glob("*.json"))
                if json_files:
                    triplet_file = json_files[0]
                    triplets = read_json(triplet_file)
                    total_before += len(triplets)

                    # Remove duplicates by converting to tuples and back
                    unique_triplets_set = {
                        tuple(sorted(d.items())) for d in triplets
                    }
                    unique_triplets_list = [dict(t) for t in unique_triplets_set]
                    total_after += len(unique_triplets_list)

                    # Save unified triplets back to the file
                    save_dict_to_json(unique_triplets_list, triplet_file)

                    if len(triplets) != len(unique_triplets_list):
                        logger.info(f"Paper {paper_dir.name}: Removed {len(triplets) - len(unique_triplets_list)} duplicates ({len(triplets)} -> {len(unique_triplets_list)})")

        logger.info(f"Unification complete: {total_before} -> {total_after} triplets (removed {total_before - total_after} duplicates)")
    else:
        logger.info("No papers processed, skipping unification step")

    # Create unified normalized triplets file
    normalized_strings_triplets = normalize_strings_triplets(normalized_triplets)
    unique_triplets = extract_unique_triplets_from_normalized_triplet_file(
        normalized_strings_triplets
    )

    normalized_triplets_file = output_dir / "normalized_triplets.json"
    save_dict_to_json(unique_triplets, normalized_triplets_file)

    logger.info(f"Step 2 completed. Output saved to: {output_dir}")
    logger.info(f"Total unique normalized triplets: {len(unique_triplets)}")

    return output_dir


# ============================================================================
# STEP 3: TDMR EXTRACTION
# ============================================================================

def extract_tdmr_results(
    normalized_triplets_dir: Path,
    tables_dir: Path,
    output_dir: Path,
    model_name: str = "openai-gpt-oss-120b",
) -> Path:
    """
    Extract TDMR results from tables using normalized triplets.

    Args:
        normalized_triplets_dir: Directory with normalized triplets
        tables_dir: Directory containing extracted tables (result_dict.json files)
        output_dir: Directory to save TDMR extraction results
        model_name: LLM model name to use

    Returns:
        Path to the output directory with TDMR results
    """
    logger.info("=" * 80)
    logger.info("STEP 3: TDMR EXTRACTION")
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

    for normalized_paper_path in tqdm(papers_with_normalized_triplets, desc="Extracting TDMR"):
        paper_name = normalized_paper_path.name
        logger.info(f"Processing paper: {paper_name}")

        if paper_name in already_processed_files:
            logger.info(f"Paper {paper_name} already processed, skipping...")
            continue

        # Create output directory for this paper
        paper_output_dir = output_dir / paper_name
        create_dir_if_not_exists(paper_output_dir)

        # Check if tables exist for this paper
        tables_paper_dir = tables_dir / paper_name
        if not tables_paper_dir.exists():
            logger.warning(f"No tables directory found for {paper_name} in {tables_dir}")
            continue

        # Load extracted tables
        tables_result_file = tables_paper_dir / "result_dict.json"
        if not tables_result_file.exists():
            logger.warning(f"No result_dict.json found for {paper_name}")
            continue

        try:
            extracted_tables_with_captions = read_json(tables_result_file)

            # Run TDMR extraction
            extract_tdmr(
                extracted_triplet_path_dir=str(normalized_paper_path),
                extracted_tables_dict_object=extracted_tables_with_captions,
                tdmr_extraction_dir=str(paper_output_dir),
            )

        except Exception as e:
            logger.error(f"Error processing {paper_name}: {str(e)}")
            continue

    logger.info(f"Step 3 completed. Output saved to: {output_dir}")
    return output_dir


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _is_step_complete(step_dir: Path, markdown_files_dir: Path) -> bool:
    """
    Check if a pipeline step has been completed for ALL papers.

    A step is considered complete if:
    - The directory exists
    - ALL papers from markdown_files_dir have been processed
    - Each paper directory contains output JSON files

    Args:
        step_dir: Directory for the pipeline step
        markdown_files_dir: Directory containing markdown files/papers

    Returns:
        True if ALL papers have been processed, False otherwise
    """
    if not step_dir.exists():
        return False

    # Get all paper names from markdown directory
    expected_papers = set()
    for paper_path in markdown_files_dir.iterdir():
        if paper_path.is_dir():
            expected_papers.add(paper_path.name)
        elif paper_path.suffix == ".md":
            expected_papers.add(paper_path.stem)

    if not expected_papers:
        return False

    # Get all processed papers from step directory
    processed_papers = set()
    for paper_dir in step_dir.iterdir():
        if paper_dir.is_dir():
            # Check if this paper has output files
            json_files = list(paper_dir.glob("*.json"))
            if json_files:
                processed_papers.add(paper_dir.name)

    # Check if all expected papers have been processed
    missing_papers = expected_papers - processed_papers

    if missing_papers:
        logger.debug(f"Step {step_dir.name} missing papers: {missing_papers}")
        return False

    return True


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline(
    true_dataset_path: str,
    base_output_dir: str,
    pdf_files_dir: Optional[str] = None,
    markdown_files_dir: Optional[str] = None,
    tables_dir: Optional[str] = None,
    model_name: str = "openai-gpt-oss-120b",
    keys_to_normalize: Optional[set] = None,
    chunk_size: int = 5000,
    resume_from_dir: Optional[str] = None,
):
    """
    Run the complete TDMR extraction pipeline.

    Args:
        true_dataset_path: Path to ground truth dataset JSON
        base_output_dir: Base directory for all outputs
        pdf_files_dir: Directory containing PDF files (optional, primary input)
        markdown_files_dir: Directory containing markdown files (optional, will be created from PDFs if not provided)
        tables_dir: Directory containing extracted tables (optional, will be created from PDFs if not provided)
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

        # Convert PDFs to markdown if markdown_files_dir not specified
        if not markdown_files_dir:
            logger.info("=" * 80)
            logger.info("STEP 0a: Converting PDFs to Markdown")
            logger.info("=" * 80)

            # Create markdown directory next to PDFs
            markdown_output_dir = pdf_files_path / "markdowns"
            create_dir_if_not_exists(markdown_output_dir)

            # Convert PDFs to markdown
            from src.parsers.docling_parsers import convert_pdf_into_md_using_docling

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
                    temp_md_path = convert_pdf_into_md_using_docling(str(pdf_file))

                    # Move to output directory
                    import shutil
                    shutil.move(temp_md_path, md_file_path)
                    logger.info(f"Successfully converted {pdf_file.name}")
                except Exception as e:
                    logger.error(f"Error converting {pdf_file.name}: {str(e)}")
                    continue

            markdown_files_dir = str(markdown_output_dir)
            logger.info(f"Markdown files created in: {markdown_files_dir}")

        # Extract tables from PDFs if tables_dir not specified
        if not tables_dir:
            logger.info("=" * 80)
            logger.info("STEP 0b: Extracting Tables from PDFs")
            logger.info("=" * 80)

            # Create tables directory next to PDFs
            tables_output_dir = pdf_files_path / "tables"
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

            tables_dir = str(tables_output_dir)
            logger.info(f"Tables extracted to: {tables_dir}")

    # Validate that we have the required directories
    if not markdown_files_dir:
        raise ValueError("Either markdown_files_dir or pdf_files_dir must be provided")
    if not tables_dir:
        raise ValueError("Either tables_dir or pdf_files_dir must be provided")

    extraction_output_dir = base_output_path / "01_triplets_extraction"
    normalization_output_dir = base_output_path / "02_triplets_normalization"
    tdmr_output_dir = base_output_path / "03_tdmr_extraction"

    # Detect which steps have been completed
    markdown_files_path = Path(markdown_files_dir)
    step1_complete = _is_step_complete(extraction_output_dir, markdown_files_path)
    step2_complete = _is_step_complete(normalization_output_dir, markdown_files_path)
    step3_complete = _is_step_complete(tdmr_output_dir, markdown_files_path)

    logger.info("=" * 80)
    logger.info("STARTING COMPLETE TDMR EXTRACTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Base output directory: {base_output_path}")

    if resume_from_dir:
        logger.info(f"Resume mode: Analyzing existing directory")
        logger.info(f"  Step 1 (Triplet Extraction): {'COMPLETE' if step1_complete else 'INCOMPLETE'}")
        logger.info(f"  Step 2 (Normalization): {'COMPLETE' if step2_complete else 'INCOMPLETE'}")
        logger.info(f"  Step 3 (TDMR Extraction): {'COMPLETE' if step3_complete else 'INCOMPLETE'}")

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

        # Step 3: Extract TDMR results
        if step3_complete and resume_from_dir:
            logger.info("=" * 80)
            logger.info("STEP 3: SKIPPING (Already completed)")
            logger.info("=" * 80)
        else:
            extract_tdmr_results(
                normalized_triplets_dir=normalization_output_dir,
                tables_dir=Path(tables_dir),
                output_dir=tdmr_output_dir,
                model_name=model_name,
            )

            # Create consolidated result files
            logger.info("Creating consolidated result files...")
            create_one_result_file(tdmr_output_dir)
            create_one_result_file_for_evaluation_purpose(tdmr_output_dir)
            logger.info("Consolidated result files created successfully")

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"All outputs saved to: {base_output_path}")
        logger.info(f"  - Triplets extraction: {extraction_output_dir}")
        logger.info(f"  - Triplets normalization: {normalization_output_dir}")
        logger.info(f"  - TDMR extraction: {tdmr_output_dir}")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


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
        model_name: LLM model name to use

    Returns:
        Path to the output directory with TDMR results
    """
    from src.utils import extract_tables_and_captions_from_pdf

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
                            user_prompt=EXTRACT_AUTHOR_APPROACH_FORM_SECTIONS_USER_PROMPT,
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
            ## TODO what if no model was extracted?
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

            # Extract tables from PDF
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
            from src.parsers.docling_parsers import convert_pdf_into_md_using_docling

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
                tables_output_dir=markdown_files_path
            )

            # Create consolidated result files
            logger.info("Creating consolidated result files...")
            create_one_result_file(tdmr_output_dir)
            create_one_result_file_for_evaluation_purpose(tdmr_output_dir)
            logger.info("Consolidated result files created successfully")

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
        "pdf_files_dir": "custom_dataset_papers_refined/dbpedia/LC-QuAD v1",  # Primary input: directory with PDF files
        "author_model_extraction_dir": "author_model_extraction",  # Required: directory with author model extraction results
        "markdown_files_dir": "custom_dataset_papers_refined/dbpedia/LC-QuAD v1/markdowns",  # markdown_files_dir will be auto-generated if not specified
        # papers_dir defaults to pdf_files_dir if not specified
        "true_dataset_path": "custom_dataset_papers_refined/dbpedia/LC-QuAD v1/LC-QuAD v1.json",
        "base_output_dir": "pipeline_results_with_author_approach",
        "model_name": "openai-gpt-oss-120b",
        "keys_to_normalize": {"Metric", "Dataset"},
        "chunk_size": 5000,
        "resume_from_dir": "10_01_2026"  # Optional: resume from specific directory
    }

    run_complete_pipeline_with_authors_extension(**CONFIG)