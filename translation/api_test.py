import os
from openai import AzureOpenAI
import requests
from dotenv import load_dotenv

load_dotenv()

# Read config from environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL") # e.g. "gpt-4o" or your model name

print(AZURE_OPENAI_ENDPOINT)
print(AZURE_OPENAI_API_KEY)
print(AZURE_OPENAI_MODEL)

def ask_azure_llm(user_message: str) -> str:
   """
   Send a chat completion request to Azure OpenAI using only `requests`.
   """
   if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_MODEL):
       raise RuntimeError("Azure OpenAI env vars are not set correctly.")


   url = AZURE_OPENAI_ENDPOINT
   headers = {
       "Content-Type": "application/json",
       "api-key": AZURE_OPENAI_API_KEY,
   }


   payload = {
       'model': AZURE_OPENAI_MODEL,
       "input": [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": user_message},
       ],
       "temperature": 0.7,
       "max_output_tokens": 256,
   }


   response = requests.post(url, headers=headers, json=payload)
   response.raise_for_status()  # raise if HTTP error


   data = response.json()
   return data['output'][0]['content'][0]['text']


if __name__ == "__main__":
   reply = ask_azure_llm("Say hi from Azure OpenAI in one sentence.")
   print("Model reply:", reply)
