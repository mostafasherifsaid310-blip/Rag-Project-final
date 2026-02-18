from hf_api import query_hf_chat_model_stream
from memory import update_memory, get_memory_context
from rag_pipeline import rag_with_citations
from ingestion import uploaded_text_memory

def hybrid_chat(user_message):

    memory_context = get_memory_context()
    files_context = "\n".join(uploaded_text_memory)

    if len(files_context) > 3000:
        files_context = files_context[-3000:]

    # كشف اللغة
    if all(ord(c) < 128 for c in user_message.replace(" ", "")):
        lang = "English"
    else:
        lang = "Arabic"
    
    prompt = f"""
أنت مساعد ذكي.

لديك مصدران للمعلومات:

1) محتوى الملفات:
{files_context}

2) سياق المحادثة:
{memory_context}

تعليمات مهمة:

- إذا كانت الملفات تحتوي على معلومات مفيدة → استخدمها
- إذا لم تكن مفيدة → تجاهلها ورد بشكل طبيعي
- لا تقل أبداً أن المعلومة غير موجودة إلا إذا كان السؤال عن الملفات تحديداً
- رد دائماً بطريقة طبيعية ومفيدة
- **لا تقطع الجمل أو الكلمات**
- **أكمل الجملة بشكل كامل قبل أن تكتب أي جزء من الإجابة**
- إذا كنت تستخدم streaming أو توليد نص تدريجي، توقف مؤقتاً حتى تكتمل الجملة.

رسالة المستخدم:
{user_message}

أجب باللغة {lang}.
"""

    


    full_answer = ""

    for partial in query_hf_chat_model_stream(prompt):
        full_answer = partial
        yield full_answer

    update_memory(user_message, full_answer)
