from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

class ChunkingLab:
    """Compare different chunking strategies on the same document."""
    
    def __init__(self):
        self.document = self._load_sample_document()
    
    def _load_sample_document(self) -> str:
        return """
# Employee Handbook 2024

## Section 1: Company Overview
Our company was founded in 2010 with the mission to revolutionize technology.
We have grown to over 5,000 employees across 12 countries.

## Section 2: Benefits

### 2.1 Vacation Policy
Employees with less than 2 years of tenure receive 10 days of vacation per year.
Employees with 2-5 years of tenure receive 15 days of vacation per year.
Employees with more than 5 years of tenure receive 20 days of vacation per year.
All vacation requests must be submitted at least 2 weeks in advance.

### 2.2 Health Insurance
Full-time employees are eligible for health insurance starting on their first day.
We offer three plans: Basic, Standard, and Premium.
Dental and vision coverage are included in the Standard and Premium plans.

### 2.3 Retirement Benefits
The company matches 401(k) contributions up to 5% of salary.
Vesting occurs over a 4-year period (25% per year).

## Section 3: Code of Conduct

### 3.1 Workplace Behavior
Employees are expected to maintain a professional environment at all times.
Harassment, discrimination, and retaliation are strictly prohibited.

### 3.2 Data Security
All employees must complete annual security training.
Confidential data must never be shared outside the company network.

## Section 4: Remote Work Policy
Employees may work remotely up to 3 days per week.
Remote work requests must be approved by your direct manager.
A home office stipend of $500 is provided for remote work equipment.
"""
    
    def fixed_size_chunk(self, size: int = 500, overlap: int = 0) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=[""]  # Only character splitting
        )
        return splitter.split_text(self.document)
    
    def recursive_chunk(self, size: int = 500, overlap: int = 50) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(self.document)
    
    def token_chunk(self, size: int = 200, overlap: int = 30) -> list[str]:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        tokens = encoding.encode(self.document)
        
        chunks = []
        step = size - overlap
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + size]
            chunks.append(encoding.decode(chunk_tokens))
        
        return chunks
    
    def analyze_chunks(self, name: str, chunks: list[str]):
        """Analyze chunk quality."""
        print(f"\n{'='*60}")
        print(f"STRATEGY: {name}")
        print(f"{'='*60}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Avg size: {sum(len(c) for c in chunks)/len(chunks):.0f} chars")
        
        # Check for broken sentences
        broken = 0
        for chunk in chunks:
            stripped = chunk.strip()
            if stripped and not stripped[-1] in '.!?"\')}]':
                broken += 1
        
        print(f"Chunks ending mid-sentence: {broken}/{len(chunks)}")
        
        # Show first 3 chunks
        print(f"\nFirst 3 chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  [{i}] ({len(chunk)} chars): {chunk[:100]}...")
        
        # Test a specific query
        test_query = "vacation days for new employees"
        print(f"\nQuery: '{test_query}'")
        
        # Find chunk containing "less than 2 years"
        for i, chunk in enumerate(chunks):
            if "less than 2 years" in chunk:
                print(f"  ✅ Found in chunk {i}: '{chunk[:150]}...'")
                # Check if policy info is also in this chunk
                if "10 days" in chunk:
                    print(f"     ✅ Answer also in same chunk!")
                else:
                    print(f"     ❌ Answer NOT in same chunk (information fragmented!)")
                break
        else:
            print(f"  ❌ Could not find relevant chunk")
        
        return chunks
    
    def run_comparison(self):
        """Run all chunking strategies."""
        
        print("🔬 CHUNKING STRATEGY COMPARISON LAB")
        
        strategies = [
            ("Fixed 500 chars, No Overlap", lambda: self.fixed_size_chunk(500, 0)),
            ("Fixed 500 chars, 50 Overlap", lambda: self.fixed_size_chunk(500, 50)),
            ("Recursive 500 chars, 50 Overlap", lambda: self.recursive_chunk(500, 50)),
            ("Token-based 200 tokens, 30 Overlap", lambda: self.token_chunk(200, 30)),
        ]
        
        results = {}
        for name, strategy_fn in strategies:
            chunks = strategy_fn()
            results[name] = self.analyze_chunks(name, chunks)
        
        return results

# Run the lab
lab = ChunkingLab()
lab.run_comparison()