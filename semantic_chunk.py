from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class SemanticChunker:
    """
    Split text into chunks based on semantic similarity between sentences.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_chunk_sentences: int = 2,
        max_chunk_sentences: int = 10
    ):
        self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
        self.min_sentences = min_chunk_sentences
        self.max_sentences = max_chunk_sentences
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Basic sentence splitting (use spaCy for production)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks."""
        
        # Get sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= self.min_sentences:
            return [text]
        
        # Embed all sentences
        embeddings = self.model.encode(sentences)
        
        # Compute similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
            similarities.append(sim)
        
        # Find breakpoints (where similarity drops)
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                breakpoints.append(i + 1)  # Break AFTER sentence i
        
        # Build chunks
        chunks = []
        start = 0
        
        for bp in breakpoints:
            chunk_sentences = sentences[start:bp]
            if len(chunk_sentences) >= self.min_sentences:
                chunks.append(' '.join(chunk_sentences))
                start = bp
        
        # Last chunk
        if start < len(sentences):
            remaining = sentences[start:]
            if len(remaining) >= self.min_sentences or chunks:
                chunks.append(' '.join(remaining))
        
        return chunks if chunks else [text]
    
    def visualize_splits(self, text: str):
        """Visualize where semantic breaks occur."""
        sentences = self._split_sentences(text)
        embeddings = self.model.encode(sentences)
        
        print("📊 Semantic Analysis of Document:\n")
        
        for i, sentence in enumerate(sentences):
            print(f"S{i}: {sentence[:80]}...")
            
            if i < len(sentences) - 1:
                sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
                bar = "█" * int(sim * 20)
                break_marker = " ⬅️ BREAK" if sim < self.threshold else ""
                print(f"    → S{i+1} similarity: {sim:.3f} {bar}{break_marker}")
                print()

# Test
chunker = SemanticChunker(similarity_threshold=0.4)

medical_text = """
The patient is a 45-year-old male presenting with acute chest pain.
An EKG was performed immediately upon arrival.
The EKG showed ST elevation in leads II, III, and aVF.
Troponin levels were elevated at 2.5 ng/mL.
The patient was diagnosed with an acute inferior myocardial infarction.
He was started on aspirin, heparin, and referred for cardiac catheterization.
His family history is significant for heart disease.
His father had a heart attack at age 52.
The patient works as a software engineer at a tech company.
He reports high stress levels and a sedentary lifestyle.
He drinks 3 cups of coffee daily and smokes half a pack of cigarettes.
"""

chunks = chunker.split_text(medical_text)

print("📄 SEMANTIC CHUNKS:")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(chunk)

chunker.visualize_splits(medical_text)