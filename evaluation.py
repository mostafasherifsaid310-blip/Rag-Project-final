import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from chat_hybrid import hybrid_chat
from ingestion import uploaded_text_memory

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def answer_relevance(answer, question):

    q_emb = embedding_model.embed_query(question)
    a_emb = embedding_model.embed_query(answer)

    return float(cosine_similarity([q_emb], [a_emb])[0][0])

def context_relevance(answer, context):

    c_emb = embedding_model.embed_query(context)
    a_emb = embedding_model.embed_query(answer)

    return float(cosine_similarity([c_emb], [a_emb])[0][0])

def faithfulness(answer):

    context = "\n".join(uploaded_text_memory)

    if len(context) > 4000:
        context = context[-4000:]

    return context_relevance(answer, context)

def hallucination_risk(faith_score):

    if faith_score > 0.75:
        return "LOW ✅"

    elif faith_score > 0.50:
        return "MEDIUM ⚠️"

    else:
        return "HIGH 🚨"
    
def confidence_score(ans_rel, faith_score):

    score = (ans_rel * 0.4) + (faith_score * 0.6)

    return round(score * 10, 2)  # من 10

def research_evaluation(question):

    from memory import conversation_history
    conversation_history.clear()

    answer = ""
    for chunk in hybrid_chat(question):
        answer = chunk

    context = "\n".join(uploaded_text_memory)

    ans_rel = answer_relevance(answer, question)
    faith_score = faithfulness(answer)
    hallucination = hallucination_risk(faith_score)
    confidence = confidence_score(ans_rel, faith_score)

    return {
        "answer": answer,
        "answer_relevance": round(ans_rel, 3),
        "faithfulness": round(faith_score, 3),
        "hallucination_risk": hallucination,
        "confidence": confidence
    }    