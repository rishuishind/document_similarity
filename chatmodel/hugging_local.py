from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace

llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100},
        )

model = ChatHuggingFace(llm=llm)

try:
    result = model.invoke("What is the capital of India and what is the capital of Haryana?")
    print(result.content)
except Exception as e:
    print(f"The error is this {e}")

