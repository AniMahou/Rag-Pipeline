import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from typing import List, Dict
import tiktoken

class AdvancedRAGLab:
    """Demonstrate HyDE, context injection, and grounding."""
    
    def __init__(self):
        self.client = OpenAI()
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction()
        self.chroma_client = chromadb.PersistentClient(path="./advanced_rag_lab")
        self.collection = self._setup_collection()
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    
    def _setup_collection(self):
        """Create collection with sample documents."""
        try:
            self.chroma_client.delete_collection("lab_kb")
        except:
            pass
        
        collection = self.chroma_client.create_collection(
            name="lab_kb",
            embedding_function=self.ef
        )
        
        documents = [
            "Employees with less than 2 years of tenure receive 10 days of vacation per year.",
            "Employees with 2-5 years of tenure receive 15 days of vacation per year.",
            "Employees with more than 5 years of tenure receive 20 days of vacation per year.",
            "Unauthorized absences of more than 3 consecutive days will be considered job abandonment.",
            "Employees must notify their supervisor of any absence prior to their scheduled shift.",
            "Health insurance coverage begins on the first day of employment.",
            "The 401(k) plan matches employee contributions up to 5% of annual salary.",
            "Remote work is permitted up to 3 days per week with manager approval.",
        ]
        
        collection.add(
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
        
        return collection
    
    def compare_retrieval_methods(self, query: str):
        """Compare standard vs HyDE retrieval."""
        
        print("\n" + "="*60)
        print(f"COMPARING RETRIEVAL METHODS")
        print(f"Query: '{query}'")
        print("="*60)
        
        # Method 1: Standard retrieval
        print("\n📊 METHOD 1: Standard Retrieval")
        std_results = self.collection.query(query_texts=[query], n_results=3)
        for i, (doc, dist) in enumerate(zip(std_results['documents'][0], std_results['distances'][0])):
            print(f"  {i+1}. [sim={1-dist:.3f}] {doc}")
        
        # Method 2: HyDE retrieval
        print("\n📊 METHOD 2: HyDE Retrieval")
        hypothetical = generate_hypothetical_answer(query, self.client)
        print(f"  Hypothetical answer: {hypothetical[:120]}...")
        
        hyde_results = self.collection.query(query_texts=[hypothetical], n_results=3)
        for i, (doc, dist) in enumerate(zip(hyde_results['documents'][0], hyde_results['distances'][0])):
            print(f"  {i+1}. [sim={1-dist:.3f}] {doc}")
    
    def compare_grounding_techniques(self, query: str):
        """Compare different grounding prompts."""
        
        # Get context
        results = self.collection.query(query_texts=[query], n_results=3)
        context = "\n\n".join(results['documents'][0])
        
        print("\n" + "="*60)
        print(f"COMPARING GROUNDING TECHNIQUES")
        print(f"Query: '{query}'")
        print("="*60)
        
        # Technique 1: No grounding
        prompt_no_ground = f"""Context: {context}
Question: {query}
Answer:"""
        
        # Technique 2: Strict grounding
        prompt_strict = GROUNDING_PROMPT.format(context=context, query=query)
        
        # Technique 3: Quote verification
        prompt_quotes = GROUNDING_WITH_QUOTES.format(context=context, query=query)
        
        techniques = {
            "No Grounding": prompt_no_ground,
            "Strict Grounding": prompt_strict,
            "Quote Verification": prompt_quotes
        }
        
        for name, prompt in techniques.items():
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            answer = response.choices[0].message.content
            
            # Check if answer contains unsupported info
            unsupported = detect_unsupported_claims(answer, context)
            
            print(f"\n📊 {name}:")
            print(f"   Answer: {answer[:150]}...")
            if unsupported:
                print(f"   ⚠️ Potentially unsupported: {unsupported}")
            else:
                print(f"   ✅ All claims supported by context")

# Run the lab
lab = AdvancedRAGLab()

# Test with informal query
lab.compare_retrieval_methods("what happens if I stop showing up to work")

# Test grounding
lab.compare_grounding_techniques("How many vacation days do new employees get?")