import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFChunker:
    """
    Extract text from PDF and chunk it preserving page boundaries.
    """
    
    def __init__(self, max_chunk_size: int = 500, preserve_pages: bool = True):
        self.max_chunk_size = max_chunk_size
        self.preserve_pages = preserve_pages
    
    def extract_pages_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from a PDF file.
        Returns list of {"page_number": int, "text": str}
        """
        pages = []
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Try multiple PDF extraction libraries
        try:
            # Method 1: pdfplumber (best for formatted text)
            import pdfplumber
            print(f"📄 Extracting with pdfplumber: {pdf_path}")
            
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append({
                            "page_number": i + 1,
                            "text": text.strip(),
                            "source": os.path.basename(pdf_path)
                        })
        
        except ImportError:
            try:
                # Method 2: PyPDF2 (lighter weight)
                from PyPDF2 import PdfReader
                print(f"📄 Extracting with PyPDF2: {pdf_path}")
                
                reader = PdfReader(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append({
                            "page_number": i + 1,
                            "text": text.strip(),
                            "source": os.path.basename(pdf_path)
                        })
            
            except ImportError:
                try:
                    # Method 3: pymupdf (fastest)
                    import fitz  # pymupdf
                    print(f"📄 Extracting with PyMuPDF: {pdf_path}")
                    
                    doc = fitz.open(pdf_path)
                    for i, page in enumerate(doc):
                        text = page.get_text()
                        if text and text.strip():
                            pages.append({
                                "page_number": i + 1,
                                "text": text.strip(),
                                "source": os.path.basename(pdf_path)
                            })
                    doc.close()
                
                except ImportError:
                    raise ImportError(
                        "No PDF library found. Install one:\n"
                        "  pip install pdfplumber\n"
                        "  pip install PyPDF2\n"
                        "  pip install pymupdf"
                    )
        
        print(f"✅ Extracted {len(pages)} pages from {os.path.basename(pdf_path)}")
        return pages
    
    def chunk(self, pages: List[Dict]) -> List[Dict]:
        """
        Chunk PDF pages.
        
        Args:
            pages: List of {"page_number": int, "text": str}
        """
        chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for page in pages:
            page_text = page["text"]
            page_num = page["page_number"]
            source = page.get("source", "document.pdf")
            
            # If page fits in one chunk
            if len(page_text) <= self.max_chunk_size:
                chunks.append({
                    "text": page_text,
                    "metadata": {
                        "page": page_num,
                        "source": source,
                        "total_pages": len(pages)
                    }
                })
            else:
                # Split page into sub-chunks
                sub_chunks = splitter.split_text(page_text)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        "text": sub_chunk,
                        "metadata": {
                            "page": page_num,
                            "sub_chunk": i + 1,
                            "total_sub_chunks": len(sub_chunks),
                            "source": source,
                            "total_pages": len(pages)
                        }
                    })
        
        print(f"✅ Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Complete pipeline: Extract → Chunk
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Step 1: Extract pages
        pages = self.extract_pages_from_pdf(pdf_path)
        
        if not pages:
            print(f"⚠️ No text extracted from {pdf_path}")
            return []
        
        # Step 2: Chunk pages
        chunks = self.chunk(pages)
        
        return chunks
    
    def save_chunks_to_file(self, chunks: List[Dict], output_path: str = None):
        """Save chunks to a text file for inspection."""
        if output_path is None:
            output_path = "pdf_chunks_output.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"PDF Chunks - Total: {len(chunks)}\n")
            f.write("="*60 + "\n\n")
            
            for i, chunk in enumerate(chunks):
                meta = chunk["metadata"]
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(f"Page: {meta.get('page', 'N/A')}\n")
                f.write(f"Source: {meta.get('source', 'N/A')}\n")
                if 'sub_chunk' in meta:
                    f.write(f"Sub-chunk: {meta['sub_chunk']}/{meta.get('total_sub_chunks', '?')}\n")
                f.write(f"\n{chunk['text']}\n")
                f.write("-"*40 + "\n\n")
        
        print(f"✅ Saved chunks to {output_path}")
        return output_path
    
    def print_chunks_preview(self, chunks: List[Dict], num_chunks: int = 3):
        """Print a preview of the first few chunks."""
        print(f"\n📊 PREVIEW (showing {min(num_chunks, len(chunks))} of {len(chunks)} chunks):")
        print("="*60)
        
        for i, chunk in enumerate(chunks[:num_chunks]):
            meta = chunk["metadata"]
            print(f"\n📄 Chunk {i+1}:")
            print(f"   Page: {meta.get('page', 'N/A')}")
            print(f"   Source: {meta.get('source', 'N/A')}")
            print(f"   Text length: {len(chunk['text'])} characters")
            print(f"   Preview: {chunk['text'][:200]}...")
            print("-"*40)


# ============================================
# USAGE - Run this to process your PDF
# ============================================

if __name__ == "__main__":
    # Your PDF file (same directory)
    pdf_filename = "tabib.pdf"  # Change this to your PDF name
    
    # Create chunker
    chunker = PDFChunker(max_chunk_size=500, preserve_pages=True)
    
    # Process the PDF
    print("🚀 Starting PDF processing...")
    print(f"📂 Looking for: {pdf_filename}")
    print(f"📂 Current directory: {os.getcwd()}")
    
    try:
        chunks = chunker.process_pdf(pdf_filename)
        
        # Preview results
        chunker.print_chunks_preview(chunks, num_chunks=5)
        
        # Save chunks to file
        chunker.save_chunks_to_file(chunks, "tabib_chunks.txt")
        
        # Print summary
        print(f"\n📊 SUMMARY:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Total characters: {sum(len(c['text']) for c in chunks):,}")
        print(f"   Avg chunk size: {sum(len(c['text']) for c in chunks)/len(chunks):.0f} chars")
        
        # Show unique pages
        pages = set(c["metadata"]["page"] for c in chunks)
        print(f"   Pages processed: {sorted(pages)}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n📁 Files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.pdf'):
                print(f"   📄 {file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")