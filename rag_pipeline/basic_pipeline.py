# Step 1: Create sample knowledge base
import chromadb
from chromadb.utils import embedding_functions

# Setup
chroma_client = chromadb.PersistentClient(path="./first_rag")
ef = embedding_functions.SentenceTransformerEmbeddingFunction()

collection = chroma_client.create_collection(
    name="company_policies",
    embedding_function=ef
)

# Sample documents
documents = [
    "Employees with less than 2 years of tenure receive 10 days of vacation per year.",
    "Employees with 2-5 years of tenure receive 15 days of vacation per year.",
    "Employees with more than 5 years of tenure receive 20 days of vacation per year.",
    "Vacation requests must be submitted at least 2 weeks in advance.",
    "Sick leave: All employees receive 5 sick days per year.",
    "Remote work: Employees may work remotely up to 3 days per week.",
    "Health insurance starts on your first day of employment.",
    "The company matches 401(k) contributions up to 5% of salary.",
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"✅ Indexed {len(documents)} documents")

# Step 2: Query
query = "How many vacation days do I get if I just started?"

results = collection.query(query_texts=[query], n_results=3)

print(f"\n🔍 Query: {query}")
print("Top 3 results:")
for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"  {i+1}. [sim={1-dist:.3f}] {doc}")

# Step 3: Generate answer with OpenAI
from openai import OpenAI
client = OpenAI()

context = "\n".join(results['documents'][0])

prompt = f"""Answer the question based ONLY on the context below.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0
)
print(f"\n🤖 Final Answer: {response.choices[0].message.content}")