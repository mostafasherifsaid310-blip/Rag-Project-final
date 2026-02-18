import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()   # يحمل متغيرات .env
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "https://router.huggingface.co/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

MODEL_NAME = "deepseek-ai/DeepSeek-V3:novita"


def query_hf_chat_model_stream(prompt, max_tokens=2000):

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "stream": True   # 🔥 هذا هو المهم
    }

    response = requests.post(
        MODEL_ID,
        headers=HEADERS,
        json=payload,
        stream=True      # 🔥 مهم أيضاً
    )

    if response.status_code != 200:
        yield f"Error {response.status_code}: {response.text}"
        return

    full_text = ""

    for line in response.iter_lines():
        if not line:
            continue

        decoded_line = line.decode("utf-8")

        # تنسيق HuggingFace Streaming
        if decoded_line.startswith("data:"):
            decoded_line = decoded_line.replace("data:", "").strip()

        if decoded_line == "[DONE]":
            break

        try:
            data = json.loads(decoded_line)

            if "choices" in data:
                delta = data["choices"][0].get("delta", {})

                if "content" in delta:
                    token = delta["content"]
                    full_text += token
                    yield full_text

        except:
            continue
