import os
from pathlib import Path

from src.parsers.marker_parser import parse_pdf_with_marker
from src.parsers.parser import parse_pdf_with_tabula
from src.utils import (create_dir_if_not_exists, extract_tables_from_markdown,
                       saving_list_of_dfs)

FILE_PATH = "/Users/Michal/Dokumenty_mac/MasterThesis/master_thesis/leaderboard-generation-papers/1603.01354.pdf"
OUTPUT_DIR = "various_tables_extraction_approaches_summary_dir"

create_dir_if_not_exists(Path(OUTPUT_DIR))

# 1. Tabula
parsed_tables_tabula = parse_pdf_with_tabula(FILE_PATH)
tabula_output_dir = os.path.join(OUTPUT_DIR, "tabula")
create_dir_if_not_exists(Path(tabula_output_dir))
saving_list_of_dfs(parsed_tables_tabula, Path(tabula_output_dir))

# 2. Marker + Regex
marker_output_dir = f"{OUTPUT_DIR}/marker_output"
market_parsed_tables_output_dir = os.path.join(OUTPUT_DIR, "marker")
create_dir_if_not_exists(Path(market_parsed_tables_output_dir))

create_dir_if_not_exists(Path(marker_output_dir))

# Extracts markdown file out of PDF
markdown_file_path = parse_pdf_with_marker(str(FILE_PATH), marker_output_dir)

extracted_tables_with_markdown = extract_tables_from_markdown(markdown_file_path)
saving_list_of_dfs(
    extracted_tables_with_markdown, Path(market_parsed_tables_output_dir)
)

# 3. Using LLM section based approach
