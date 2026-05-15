import chromadb
from openai import OpenAI
from typing import List, Dict, Optional
import tiktoken

class RAGPipeline:
    """
    Complete RAG pipeline with query processing.
    """
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        persist_path: str = "./rag_db"
    ):
        self.client = OpenAI()
        self.model = model
        self.embedding_model = embedding_model
        self.encoding = tiktoken.encoding_for_model(model)
        
        # Connect to vector database
        self.chroma_client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        
        self.conversation_history = []
    
    # ─── STAGE 1: QUERY PROCESSING ───
    
    def process_query(
        self,
        query: str,
        rewrite: bool = True,
        expand: bool = False
    ) -> str:
        """
        Process the raw user query for better retrieval.
        """
        # Clean
        query = self._clean_query(query)
        
        # Rewrite with conversation context
        if rewrite:
            query = self._rewrite_query(query)
        
        print(f"🔍 Processed query: {query}")
        return query
    
    def _clean_query(self, query: str) -> str:
        """Basic query cleaning."""
        import re
        
        query = query.strip()
        
        # Remove filler
        fillers = ["um,", "uh,", "like,", "you know,", "i was wondering"]
        for filler in fillers:
            query = query.lower().replace(filler, "")
        
        # Fix excessive punctuation
        query = re.sub(r'[?]{2,}', '?', query)
        
        return query.strip()
    
    def _rewrite_query(self, query: str) -> str:
        """Rewrite query with conversation context."""
        
        history_text = ""
        if self.conversation_history:
            recent = self.conversation_history[-4:]  # Last 2 exchanges
            history_text = "\n".join(
                f"{'User' if i%2==0 else 'Assistant'}: {msg}"
                for i, msg in enumerate(recent)
            )
        
        prompt = f"""Convert this question into a clear, standalone search query.
Resolve any ambiguous pronouns.

Conversation history:
{history_text if history_text else "None"}

Question: {query}

Search query:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip()
    
    # ─── STAGE 2 & 3: EMBED + RETRIEVE ───
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Embed query and retrieve relevant chunks.
        """
        # Query the vector store
        results = self.collection.query(
            query_texts=[query],
            n_results=k * 2,  # Oversample for threshold filtering
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        retrieved = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = 1 - dist  # Convert distance to similarity
            
            if similarity >= threshold:
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": similarity
                })
        
        # Return top-k
        return retrieved[:k]
    
    # ─── STAGE 4: CONTEXT ASSEMBLY ───
    
    def build_context(
        self,
        chunks: List[Dict],
        max_tokens: int = 3000
    ) -> str:
        """
        Assemble retrieved chunks into a context for the LLM.
        """
        context_parts = []
        current_tokens = 0
        
        for i, chunk in enumerate(chunks):
            chunk_tokens = len(self.encoding.encode(chunk["text"]))
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            # Add with source citation
            source = chunk.get("metadata", {}).get("source", "Unknown")
            page = chunk.get("metadata", {}).get("page", "")
            
            context_parts.append(
                f"[Source: {source}"
                + (f", Page {page}]" if page else "]")
                + f"\n{chunk['text']}"
            )
            
            current_tokens += chunk_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    # ─── STAGE 5: GENERATION ───
    
    def generate(
        self,
        query: str,
        context: str,
        temperature: float = 0.0
    ) -> str:
        """
        Generate answer from context.
        """
        
        system_prompt = """You are a precise information assistant.

RULES:
1. Answer ONLY using the provided context.
2. If the answer is not in the context, say: "I don't have that information in my knowledge base."
3. When you use information from the context, mention the source.
4. Be concise but complete.
5. Do not use outside knowledge."""

        user_prompt = f"""CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    # ─── FULL PIPELINE ───
    
    def ask(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5,
        rewrite: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline: Process → Retrieve → Generate.
        """
        print(f"\n{'='*60}")
        print(f"💬 USER: {query}")
        print('='*60)
        
        # Stage 1: Process query
        processed_query = self.process_query(query, rewrite=rewrite)
        
        # Stage 2 & 3: Retrieve
        chunks = self.retrieve(processed_query, k=k, threshold=threshold)
        
        if not chunks:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": [],
                "processed_query": processed_query
            }
        
        print(f"📊 Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  {i+1}. [sim={chunk['similarity']:.3f}] {chunk['text'][:80]}...")
        
        # Stage 4: Build context
        context = self.build_context(chunks)
        
        # Stage 5: Generate
        answer = self.generate(processed_query, context)
        
        # Update conversation history
        self.conversation_history.append(query)
        self.conversation_history.append(answer)
        
        # Extract sources
        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Unknown")
            page = meta.get("page", "")
            sources.append(f"{source}" + (f" (p.{page})" if page else ""))
        
        result = {
            "answer": answer,
            "sources": list(set(sources)),
            "processed_query": processed_query,
            "num_chunks": len(chunks)
        }
        
        print(f"\n🤖 ANSWER: {answer}")
        
        return result

# ─── USAGE ───

# Initialize pipeline
rag = RAGPipeline(
    collection_name="knowledge_base",
    persist_path="./rag_db"
)

# Index some documents first (assuming you have chunks from Class 10)
# rag.collection.add(documents=chunks, ...)

# Ask a question
result = rag.ask("What is the vacation policy for new employees?")

print(f"\n📋 Final Result:")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")