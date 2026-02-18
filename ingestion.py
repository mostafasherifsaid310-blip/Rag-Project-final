import fitz
from docx import Document

# ذاكرة لتخزين نصوص الملفات (بعد تقسيمها)
uploaded_text_memory = []

def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_docx_text(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_pdf_text(file_path)
    elif file_path.endswith(".docx"):
        return extract_docx_text(file_path)
    else:
        raise ValueError("Unsupported file format")

def chunk_text(text, chunk_size=2000):
    """قسم النصوص الكبيرة إلى أجزاء صغيرة"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def upload_files(files):
    """
    رفع الملفات، تقسيمها إلى أجزاء صغيرة وتخزينها في الذاكرة
    """
    global uploaded_text_memory
    all_texts = []
    for f in files:
        try:
            text = extract_text(f.name)
            chunks = chunk_text(text)
            for chunk in chunks:
                formatted_chunk = f"--- {f.name} ---\n{chunk}"
                uploaded_text_memory.append(formatted_chunk)
            all_texts.append(f"--- {f.name} ---\n{text[:1000]}...")  # معاينة أول 1000 حرف
        except Exception as e:
            all_texts.append(f"Error in {f.name}: {str(e)}")
    return "\n\n".join(all_texts)



