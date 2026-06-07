from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 

sentence = "Delhi is the captal of India"
try:
    result = emb.embed_query(sentence)
    print(str(result))
except Exception as e:
    print(f"Error is {e}")
