from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="conversational"
        )

model = ChatHuggingFace(llm=llm)

try:
    response = model.invoke("What is the capital of India ?")
    print(response.content)
except Exception as e:
    print(f"An error occured: {e}")

