from openai import OpenAI
import numpy as np

client = OpenAI(api_key="sk-svcacct-oD2ywALJLsVNQwU3GZ-Lvl-6d53TV1hwjr7DKEfvT07SSYDzFHvmTo3Fx9gp-5KT3BlbkFJsc-MOn7QvehX14nXoONvR0tKAQ8_XBfDFywvS3CVLyL4QaCm9Pklco6v0i3hhAA")

#===== Creating an embedding with the openai api =======
# https://platform.openai.com/docs/guides/embeddings
response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-small"
)
print(response.data[0].embedding)
# ======================================================



# ==== Use case example ======
query = "Who is the greatest basketball player of all time?"

resource1 = "Lebron james is likely the greatest NBA player of all time."
resource2 = "Messi is likely the greatest soccer player of all time"
resource3 = "RAG stands for Retrieval Augmented Generation"

query_embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

resource_responses = client.embeddings.create(
            input=[resource1, resource2, resource3],
            model="text-embedding-3-small"
    ).data

resource1_embedding = resource_responses[0].embedding
resource2_embedding = resource_responses[1].embedding
resource3_embedding = resource_responses[2].embedding

query = np.array(query_embedding)

resource1_embedding = np.array(resource1_embedding)
resource2_embedding = np.array(resource2_embedding)
resource3_embedding = np.array(resource3_embedding)

def cosine_simularity(A, B):
    return np.dot(A,B) / ( np.linalg.norm(A) * np.linalg.norm(B) )

simularity1 = cosine_simularity(query_embedding, resource1_embedding)
simularity2 = cosine_simularity(query_embedding, resource2_embedding)
simularity3 = cosine_simularity(query_embedding, resource3_embedding)

print(f"Simularity of query and resource1: {simularity1}")
print(f"Simularity of query and resource2: {simularity2}")
print(f"Simularity of query and resource3: {simularity3}")


# Step 1: Embed chunks of the `messi.txt` text file
with open("messi.txt", "r") as f:
    text = f.read()
    print(text)

# Define the chunk size for splitting the text
CHUNNK_SIZE = 100
chunks = []
current_place = 0
while current_place < len(text):
    current_chunk = text[current_place : current_place + CHUNNK_SIZE]
    chunks.append(current_chunk)
    current_place += CHUNNK_SIZE
print(chunks)

# Step 2: Embed each chunk and store in a dictionary
embedded_chunks = []
for chunk in chunks:
    response = client.embeddings.create(
        input=chunk,
        model="text-embedding-3-small"
    )
    embedded_chunks.append( (chunk, response.data[0].embedding) )
print(embedded_chunks)

#Calculate cosine similarity to find relevant chunks
def cosine_simularity(A, B):
    return np.dot(A,B) / ( np.linalg.norm(A) * np.linalg.norm(B) )

question = "why was messi relocated to spain?"

question_embedding = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding
print(question_embedding)

best_cosine_similarity = 0
index = 0
for i in range(len(embedded_chunks)):
    similarity = cosine_simularity(question_embedding, embedded_chunks[i][1])
    if similarity > best_cosine_similarity:
        best_cosine_similarity = similarity
        index = i

print(f"Relevant information: {embedded_chunks[index][0]}")

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user", "content": f"User question: {question}. Potentially useful information: {embedded_chunks[index][0]}"
        }
    ]
)

# Output the response from the LLM
print(completion.choices[0].message.content)