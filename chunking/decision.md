What type of document are you chunking?
│
├─ 📝 Plain Text / Articles
│   └─ Use: RecursiveCharacterTextSplitter
│       chunk_size: 512-1024 tokens
│       overlap: 10-20%
│       separators: ["\n\n", "\n", ". ", " ", ""]
│
├─ 📄 PDF Documents
│   └─ Use: PDF-aware chunking
│       Preserve page boundaries
│       Add page metadata for citations
│       chunk_size: 500-800 tokens
│       overlap: 50-100 tokens
│
├─ 📑 Markdown / Documentation
│   └─ Use: Header-aware chunking
│       Split on ## and ### headers
│       Preserve header hierarchy
│       Keep code blocks intact
│
├─ 💻 Source Code
│   └─ Use: AST-aware chunking (functions/classes)
│       Keep imports at module level
│       Include docstrings and comments
│       chunk_size: By function, not characters
│
├─ 🌐 HTML / Web Pages
│   └─ Use: DOM-aware chunking
│       Extract from main content elements
│       Remove nav, footer, ads
│       Preserve header hierarchy
│
├─ 📊 Data-Heavy (Tables, Charts)
│   └─ Use: Multimodal chunking
│       Linearize tables for embedding
│       Store raw tables for LLM context
│       Describe images/charts with vision model
│
├─ 🔬 Scientific Papers
│   └─ Use: Section-aware chunking
│       Keep Abstract, Methods, Results separate
│       Preserve figure captions with references
│       chunk_size: Smaller for Abstract, larger for Methods
│
└─ 🏢 Enterprise Documents (Mixed)
    └─ Use: Adaptive chunking
        Detect section types automatically
        Apply different strategies per section
        Unified metadata schema