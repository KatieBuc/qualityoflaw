import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from translation.llm.azure_client import get_api_style, get_azure_client, get_azure_model


def ask_azure_llm(user_message: str) -> str:
    client = get_azure_client()
    model = get_azure_model()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]

    if get_api_style() == "responses":
        response = client.responses.create(
            model=model,
            input=messages,
            temperature=0.7,
            max_output_tokens=256,
        )
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        return response.output[0].content[0].text

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    reply = ask_azure_llm("Say hi from Azure OpenAI in one sentence.")
    print("Model reply:", reply)
