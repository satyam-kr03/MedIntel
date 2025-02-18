from unstructured.partition.auto import partition

def process_pdf(filename: str, path: str):
    raw_pdf_elements = partition(
        filename=path + filename,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )
    return raw_pdf_elements
