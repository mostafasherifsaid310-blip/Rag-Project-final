conversation_history = []


def update_memory(question, answer):
    conversation_history.append({
        "question": question,
        "answer": answer
    })


def get_memory_context(last_n=5):
    return "\n".join([
        f"سؤال: {item['question']}\nإجابة: {item['answer']}"
        for item in conversation_history[-last_n:]
    ])
