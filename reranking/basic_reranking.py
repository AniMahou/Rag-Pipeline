import cohere

co = cohere.Client(api_key="your-cohere-api-key")

def rerank_with_cohere(
    query: str,
    documents: list[str],
    top_n: int = 3,
    model: str = "rerank-english-v3.0"
) -> list[dict]:
    """
    Re-rank documents using Cohere Rerank API.
    
    Args:
        query: User question
        documents: List of document texts (from Stage 1)
        top_n: Number of documents to return
        model: Cohere rerank model
    
    Returns:
        Re-ranked list with relevance scores
    """
    
    response = co.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model=model
    )
    
    results = []
    for result in response.results:
        results.append({
            "index": result.index,
            "document": documents[result.index],
            "relevance_score": result.relevance_score
        })
    
    return results


# Usage in RAG pipeline
def two_stage_retrieval_cohere(query: str, collection, k_retrieve: int = 20, k_final: int = 3):
    """
    Two-stage retrieval: Retrieve 20, re-rank to top 3.
    """
    
    # Stage 1: Fast retrieval with embeddings
    stage1_results = collection.query(
        query_texts=[query],
        n_results=k_retrieve
    )
    
    candidates = stage1_results['documents'][0]
    
    print(f"📊 Stage 1: Retrieved {len(candidates)} candidates")
    
    # Stage 2: Re-rank with Cohere
    stage2_results = rerank_with_cohere(
        query=query,
        documents=candidates,
        top_n=k_final
    )
    
    print(f"🎯 Stage 2: Re-ranked to top {len(stage2_results)}")
    
    for i, result in enumerate(stage2_results):
        print(f"  {i+1}. [score={result['relevance_score']:.3f}] {result['document'][:80]}...")
    
    return stage2_results