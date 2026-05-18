import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from typing import List, Dict, Optional, Generator
import tiktoken
import time
import json

class ProductionRAG:
    """
    Complete production-ready RAG system with all features.
    """
    
    def __init__(
        self,
        collection_name: str = "production_kb",
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        persist_path: str = "./production_rag"
    ):
        # LLM
        self.client = OpenAI()
        self.model = model
        
        # Vector DB
        self.chroma_client = chromadb.PersistentClient(path=persist_path)
        
        # Embedding function
        try:
            self.ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name=embedding_model
            )
        except:
            self.ef = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef
        )
        
        # Token management
        self.encoding = tiktoken.encoding_for_model(model)
        
        # Conversation memory
        self.conversation_history = []
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "total_tokens": 0,
            "avg_retrieval_time_ms": 0,
            "avg_generation_time_ms": 0
        }
    
    def ask(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5,
        use_hyde: bool = False,
        stream: bool = False,
        grounding: str = "strict"
    ) -> Dict:
        """
        Complete RAG query with all options.
        
        Args:
            query: User question
            k: Number of chunks to retrieve
            threshold: Minimum similarity score
            use_hyde: Use HyDE for retrieval
            stream: Stream the response
            grounding: "strict", "quotes", or "basic"
        
        Returns:
            Dict with answer, sources, metadata
        """
        
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Stage 1: Process query
        processed_query = self._contextualize_query(query)
        
        # Stage 2: Retrieve
        retrieval_start = time.time()
        
        if use_hyde:
            chunks = self._hyde_retrieve(processed_query, k)
        else:
            chunks = self._retrieve(processed_query, k, threshold)
        
        retrieval_time = (time.time() - retrieval_start) * 1000
        self.stats["avg_retrieval_time_ms"] = (
            (self.stats["avg_retrieval_time_ms"] * (self.stats["total_queries"] - 1) + retrieval_time)
            / self.stats["total_queries"]
        )
        
        if not chunks:
            return {
                "answer": "I don't have enough information to answer that.",
                "sources": [],
                "retrieved_chunks": 0
            }
        
        # Stage 3: Build context
        context = self._build_context(chunks)
        
        # Stage 4: Generate
        gen_start = time.time()
        
        answer = self._generate(
            query=query,
            context=context,
            grounding=grounding,
            stream=stream
        )
        
        gen_time = (time.time() - gen_start) * 1000
        self.stats["avg_generation_time_ms"] = (
            (self.stats["avg_generation_time_ms"] * (self.stats["total_queries"] - 1) + gen_time)
            / self.stats["total_queries"]
        )
        
        # Stage 5: Post-process
        sources = self._extract_sources(chunks)
        
        # Update conversation
        self.conversation_history.append({
            "user": query,
            "assistant": answer if not stream else "[streamed]"
        })
        
        result = {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(chunks),
            "processed_query": processed_query,
            "timing_ms": {
                "retrieval": retrieval_time,
                "generation": gen_time,
                "total": (time.time() - start_time) * 1000
            }
        }
        
        return result
    
    def _contextualize_query(self, query: str) -> str:
        """Rewrite query with conversation context."""
        if not self.conversation_history:
            return query
        
        history_text = "\n".join(
            f"User: {ex['user']}\nAssistant: {ex['assistant'][:200]}"
            for ex in self.conversation_history[-3:]
        )
        
        prompt = f"""Rewrite this question as a standalone query.

Conversation:
{history_text}

Question: {query}

Standalone:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip()
    
    def _retrieve(self, query: str, k: int, threshold: float) -> List[Dict]:
        """Standard retrieval."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k * 2,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = 1 - dist
            if similarity >= threshold:
                chunks.append({
                    "text": doc,
                    "metadata": meta or {},
                    "similarity": similarity
                })
        
        return chunks[:k]
    
    def _hyde_retrieve(self, query: str, k: int) -> List[Dict]:
        """HyDE retrieval."""
        # Generate hypothetical answer
        hypo_prompt = f"Write a paragraph answering: {query}\n\nAnswer:"
        hypo_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": hypo_prompt}],
            temperature=0.0
        )
        hypothetical = hypo_response.choices[0].message.content
        
        # Search with hypothetical
        return self._retrieve(hypothetical, k, threshold=0.3)  # Lower threshold for HyDE
    
    def _build_context(self, chunks: List[Dict], max_tokens: int = 3000) -> str:
        """Build context with token budget."""
        parts = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = len(self.encoding.encode(chunk["text"]))
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            source = chunk["metadata"].get("source", "Unknown")
            parts.append(f"[Source: {source}]\n{chunk['text']}")
            current_tokens += chunk_tokens
        
        return "\n\n---\n\n".join(parts)
    
    def _generate(
        self,
        query: str,
        context: str,
        grounding: str = "strict",
        stream: bool = False
    ) -> str:
        """Generate answer with grounding."""
        
        if grounding == "strict":
            system = """Answer ONLY using the provided context. 
If the answer is not in the context, say: "I don't have this information." 
Do NOT use outside knowledge."""
        
        elif grounding == "quotes":
            system = """Answer using the context. 
For each claim, include a [QUOTE] from the context.
If not in context, say so."""
        
        else:
            system = "Answer based on the context when possible."
        
        user = f"""CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        if stream:
            return self._stream_generate(system, user)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
    
    def _stream_generate(self, system: str, user: str) -> Generator:
        """Stream generation."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.0,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _extract_sources(self, chunks: List[Dict]) -> List[str]:
        """Extract unique sources."""
        sources = []
        for chunk in chunks:
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", "")
            citation = source + (f" (p.{page})" if page else "")
            if citation not in sources:
                sources.append(citation)
        return sources
    
    def index_documents(self, chunks: List[Dict], batch_size: int = 100):
        """Index pre-chunked documents."""
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            self.collection.add(
                documents=[c["text"] for c in batch],
                metadatas=[c.get("metadata", {}) for c in batch],
                ids=[f"chunk_{i+j}" for j in range(len(batch))]
            )
        
        print(f"✅ Indexed {len(chunks)} chunks")
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            **self.stats,
            "collection_size": self.collection.count(),
            "conversation_turns": len(self.conversation_history)
        }

# ─── USAGE ───

# Initialize
rag = ProductionRAG(
    collection_name="company_kb",
    persist_path="./production_rag"
)

# Index documents (from your PDF chunker)
# chunks = pdf_chunker.process_pdf("handbook.pdf")
# rag.index_documents(chunks)

# Ask a question
result = rag.ask(
    query="What is the vacation policy for new employees?",
    use_hyde=False,
    grounding="strict"
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Timing: {result['timing_ms']}")

# Follow-up question (uses conversation memory)
result2 = rag.ask("How do I submit a request for that?")
print(f"Answer: {result2['answer']}")

# Check stats
print(f"\nStats: {json.dumps(rag.get_stats(), indent=2)}")