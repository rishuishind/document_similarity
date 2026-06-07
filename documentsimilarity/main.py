from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

try:
    doc_embedding = embedding.embed_documents(documents)

    user_query = input("Enter the query you want to search for: \n")
    query_embedding = embedding.embed_query(user_query)

    # Now we will find the cosine similarity between document and user query
    scores = cosine_similarity([query_embedding],doc_embedding)[0]
    index,score = max(enumerate(scores),key=lambda x:x[1])
    print(documents[index])

except Exception as e:
    print(f"Error while fetching as {e}")
