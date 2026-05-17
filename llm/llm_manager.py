import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from google.colab import userdata

# 1. Fetch your cloud token securely from Colab Secrets
try:
    hf_token = userdata.get('HF_TOKEN')
except Exception:
    raise ValueError("❌ Please set your 'HF_TOKEN' in the Colab Secrets (key icon) sidebar.")

# 2. Setup the serverless endpoint for Gemma 2 9B
llm_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=hf_token,
)

# 3. Wrap it in the chat interface
llm = ChatHuggingFace(llm=llm_endpoint)

# 4. Test the cloud connection
try:
    print("Testing connection to Gemma 2 9B via Hugging Face...")
    response = llm.invoke("Hello! Are you ready to analyze some papers?")
    print(f"\n🤖 Response:\n{response.content}")
except Exception as e:
    print(f"❌ Connection failed: {e}")