from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

chat_history = ""
model = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="conversation"
        )

llm = ChatHuggingFace(llm=model)

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break;
    response = llm.invoke(chat_history + "\n" + user_input)
    chat_history = chat_history + f"User: {user_input} \n AI: {response.content} \n"
    print("AI: ",response.content)
