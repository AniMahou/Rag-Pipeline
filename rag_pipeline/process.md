┌─────────────────────────────────────────────────────────────────────┐
│                        THE RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User: "What is the vacation policy?"                               │
│    │                                                                │
│    ▼                                                                │
│  ┌─────────────────┐                                                │
│  │ QUERY PROCESSOR │  ← Clean, rewrite, expand the query            │
│  └────────┬────────┘                                                │
│           │ "vacation policy employees less than 2 years"           │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │    EMBEDDER     │  ← Convert query to vector                     │
│  └────────┬────────┘                                                │
│           │ [0.12, -0.45, 0.78, ...]                                │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │  VECTOR SEARCH  │  ← Find similar chunks in database             │
│  └────────┬────────┘                                                │
│           │ Chunk 42 (0.92), Chunk 17 (0.87), Chunk 99 (0.76)      │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │    RETRIEVER    │  ← Fetch full text of top chunks               │
│  └────────┬────────┘                                                │
│           │ "Employees with less than 2 years tenure..."            │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │ CONTEXT BUILDER │  ← Format chunks into prompt                   │
│  └────────┬────────┘                                                │
│           │ [System: Answer from context] [Context: ...] [Query]    │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │   LLM GENERATOR │  ← Generate answer from context                │
│  └────────┬────────┘                                                │
│           │ "Employees with less than 2 years of tenure..."         │
│           ▼                                                         │
│  User: "Employees with less than 2 years of tenure receive         │
│         10 days of vacation per year."                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘