import fitz


def extract_text_from_pdf(pdf_file):
    """
    Extract text from an uploaded PDF resume.

    Args:
        pdf_file: Uploaded PDF file from Streamlit.

    Returns:
        Extracted resume text as a string.
    """

    try:
        pdf_bytes = pdf_file.getvalue()

        with fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as document:

            extracted_text = []

            for page in document:
                extracted_text.append(page.get_text())

        text = "\n".join(extracted_text).strip()

        if not text:
            return None, (
                "No readable text was found. "
                "The PDF may be scanned or image-based."
            )

        return text, None

    except Exception:
        return None, (
            "Unable to read this PDF. "
            "Please upload a valid, readable resume."
        )