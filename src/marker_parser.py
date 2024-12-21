import os
from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter
from src.logger import logger
from src.utils import create_dir_if_not_exists, run_bash_command


def split_pdf_files(folder_in: str, folder_out: str) -> None:
    """
    Splits each page of PDF documents in a specified input folder into separate PDF files in an output folder.

    Args:
        folder_in (str): Path to the input directory containing PDF files.
        folder_out (str): Path to the output directory where split PDF files will be saved.

    Note:
        If an error occurs during the splitting of a PDF, the function prints the file path that caused the error.
    """

    os.makedirs(folder_out, exist_ok=True)

    for fname in os.listdir(folder_in):
        fpath = f"{folder_in}/{fname}"
        try:
            inputpdf = PdfReader(open(fpath, "rb"))
            for i in range(len(inputpdf.pages)):
                output = PdfWriter()
                output.add_page(inputpdf.pages[i])
                with open(
                    f'{folder_out}/{fname.split(".")[0].replace(" ", "-")}__{i}.pdf',
                    "wb",
                ) as outputStream:
                    output.write(outputStream)
        except Exception as e:
            logger.error(e)
            print(fpath)


def concat_markdown_files(dir_in: str, out_dir: str) -> None:
    """
    Concatenates markdown files from all unique base files in the input directory into the output directory.

    Args:
        dir_in (str): Input directory containing markdown files and associated data.
        out_dir (str): Output directory where concatenated files will be stored.
    """

    os.makedirs(out_dir, exist_ok=True)
    for file in define_unique_files(dir_in):
        concat_markdown_files(dir_in, out_dir, file)


# TODO add logic with splitting PDF file!
def parse_pdf_with_marker(pdf_file_path: str, output_dir: str) -> str:
    if Path(output_dir).exists():
        logger.info(f"Provided path: {output_dir} exists, skipping running marker ...")
    else:
        logger.info(f"Running marker for this PDF: {pdf_file_path}")
        create_dir_if_not_exists(Path(output_dir))
        run_bash_command(f"marker_single {pdf_file_path} --output_dir {output_dir}")
        logger.info("Running marker for this PDF completed")

    markdown_file_path = os.path.join(
        output_dir, Path(pdf_file_path).stem, Path(pdf_file_path).stem + ".md"
    )
    return markdown_file_path
