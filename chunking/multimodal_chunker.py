#!/usr/bin/env python3
"""
Multimodal Chunking Demo - Process PDF with Images, Tables, and Diagrams
Supports: Text chunks, Table-aware chunks, Diagram-aware chunks
"""

import os
import sys
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

# PDF Processing
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("⚠️ PyMuPDF not installed. Run: pip install PyMuPDF")

# Image processing
try:
    from PIL import Image
    import base64
    from io import BytesIO
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️ Pillow not installed. Run: pip install Pillow")

# Try to import optional vision model
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️ sentence-transformers not installed. Run: pip install sentence-transformers")

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: BASE CHUNKING STRATEGIES
# ============================================================================

class RecursiveChunker:
    """Standard recursive character chunking."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str) -> List[Dict]:
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "metadata": {"chunking_strategy": "recursive"}
            })
            
            start = end - self.chunk_overlap
        
        return chunks


# ============================================================================
# SECTION 2: TABLE-AWARE CHUNKING
# ============================================================================

class TableAwareChunker:
    """
    Chunk documents with special handling for tables.
    Tables are linearized for embedding and kept raw for LLM context.
    """
    
    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
    
    def chunk_pdf_with_tables(self, pdf_path: str) -> List[Dict]:
        """Extract tables from PDF and chunk them intelligently."""
        
        if not HAS_FITZ:
            return [{"text": "PyMuPDF not available", "metadata": {"error": True}}]
        
        chunks = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc, start=1):
            # Extract tables from page
            tables = page.find_tables()
            
            if tables.tables:
                for table_idx, table in enumerate(tables.tables):
                    table_data = self._extract_table_data(table)
                    table_chunk = self._process_table(
                        table_data, 
                        caption=f"Page {page_num}, Table {table_idx + 1}"
                    )
                    table_chunk["metadata"]["page"] = page_num
                    table_chunk["metadata"]["chunking_strategy"] = "table_aware"
                    chunks.append(table_chunk)
            
            # Also extract regular text from page
            page_text = page.get_text()
            if page_text.strip():
                text_chunks = self._chunk_text(page_text, page_num)
                chunks.extend(text_chunks)
        
        doc.close()
        return chunks
    
    def _extract_table_data(self, table) -> List[List[str]]:
        """Extract table content as list of rows."""
        try:
            # PyMuPDF table extraction
            data = table.extract()
            return data if data else []
        except:
            return []
    
    def _process_table(self, table_data: List[List[str]], caption: str = "") -> Dict:
        """Convert table to embedding-friendly and LLM-friendly formats."""
        
        if not table_data or len(table_data) < 2:
            return {"text": "", "raw_table": "", "metadata": {"type": "table_empty"}}
        
        headers = table_data[0] if table_data else []
        rows = table_data[1:] if len(table_data) > 1 else []
        
        # Strategy 1: Linearize for embedding (searchable)
        linearized = f"Table: {caption}\n"
        row_count = 0
        
        for row in rows[:10]:  # Limit for embedding size
            if len(row) > 0:
                row_text = "; ".join(
                    f"{headers[i] if i < len(headers) else f'Col{i}'}: {cell}" 
                    for i, cell in enumerate(row) 
                    if cell and str(cell).strip()
                )
                if row_text:
                    linearized += f"- {row_text}\n"
                    row_count += 1
        
        # Strategy 2: Raw markdown table for LLM context
        md_table = f"**{caption}**\n\n"
        if headers:
            md_table += "| " + " | ".join(str(h) for h in headers) + " |\n"
            md_table += "|" + "|".join(["---" for _ in headers]) + "|\n"
        
        for row in rows[:20]:  # Show up to 20 rows in raw format
            if len(row) > 0:
                md_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        return {
            "text": linearized if linearized else f"Table with {len(rows)} rows",  # This gets embedded
            "raw_table": md_table,  # This gets injected into LLM context
            "metadata": {
                "type": "table",
                "caption": caption,
                "rows": len(rows),
                "columns": len(headers),
                "display_rows": row_count
            }
        }
    
    def _chunk_text(self, text: str, page_num: int) -> List[Dict]:
        """Normal text chunking for non-table content."""
        # Simple recursive splitting
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1
            if current_size > self.max_chunk_size and current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "raw_table": None,
                    "metadata": {
                        "type": "text",
                        "page": page_num,
                        "chunking_strategy": "table_aware_text"
                    }
                })
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "raw_table": None,
                "metadata": {
                    "type": "text", 
                    "page": page_num,
                    "chunking_strategy": "table_aware_text"
                }
            })
        
        return chunks


# ============================================================================
# SECTION 3: MULTIMODAL CHUNKER (Images + Text)
# ============================================================================

class MultimodalChunker:
    """
    Chunk documents containing images and text together.
    Keeps images with their surrounding context.
    """
    
    def __init__(self, max_chunk_size: int = 500, vision_model=None):
        self.max_chunk_size = max_chunk_size
        self.vision_model = vision_model
    
    def chunk_pdf_multimodal(self, pdf_path: str) -> List[Dict]:
        """Extract and chunk PDF with images preserved."""
        
        if not HAS_FITZ:
            return [{"text": "PyMuPDF not available", "metadata": {"error": True}}]
        
        chunks = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc, start=1):
            # Get page text
            page_text = page.get_text()
            
            # Extract images from page
            images = page.get_images(full=True)
            
            # Build elements for this page
            elements = []
            
            # Add text first
            if page_text.strip():
                elements.append({"type": "text", "content": page_text})
            
            # Add images
            for img_idx, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save image temporarily or encode as base64
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    elements.append({
                        "type": "image",
                        "content": image_b64,
                        "caption": f"Image {img_idx + 1} on page {page_num}"
                    })
                except Exception as e:
                    print(f"   ⚠️ Could not extract image: {e}")
            
            # Chunk the elements for this page
            page_chunks = self._chunk_elements(elements, page_num)
            chunks.extend(page_chunks)
        
        doc.close()
        return chunks
    
    def _chunk_elements(self, elements: List[Dict], page_num: int) -> List[Dict]:
        """Chunk a list of text and image elements."""
        
        chunks = []
        current_chunk = {"text": "", "images": [], "captions": [], "image_descriptions": []}
        current_size = 0
        
        for element in elements:
            if element["type"] == "text":
                text = element["content"]
                new_size = current_size + len(text)
                
                if new_size > self.max_chunk_size and current_chunk["text"]:
                    # Save current chunk
                    chunks.append(self._finalize_chunk(current_chunk, page_num))
                    current_chunk = {"text": text, "images": [], "captions": [], "image_descriptions": []}
                    current_size = len(text)
                else:
                    if current_chunk["text"]:
                        current_chunk["text"] += "\n" + text
                    else:
                        current_chunk["text"] = text
                    current_size = new_size
            
            elif element["type"] == "image":
                image_data = element.get("content")
                caption = element.get("caption", "")
                
                # Describe image if we have a vision model
                description = self._describe_image(image_data) if self.vision_model else ""
                
                if description:
                    current_chunk["text"] += f"\n[Image Description: {description}]"
                    current_chunk["image_descriptions"].append(description)
                
                current_chunk["images"].append(image_data[:100] if image_data else None)  # Store preview
                current_chunk["captions"].append(caption)
        
        # Don't forget the last chunk
        if current_chunk["text"] or current_chunk["images"]:
            chunks.append(self._finalize_chunk(current_chunk, page_num))
        
        return chunks
    
    def _describe_image(self, image_data: str) -> str:
        """Describe image using vision model or fallback."""
        # This is a placeholder - in production, use GPT-4V or similar
        # For demo, return a generic description
        return "Diagram or chart visual content"
    
    def _finalize_chunk(self, chunk: Dict, page_num: int) -> Dict:
        """Combine text and image descriptions into a single chunk."""
        
        enriched_text = chunk["text"]
        
        for caption in chunk["captions"]:
            if caption and caption not in enriched_text:
                enriched_text += f"\n[Figure: {caption}]"
        
        return {
            "text": enriched_text,
            "raw_table": None,
            "has_images": len(chunk["images"]) > 0,
            "metadata": {
                "type": "multimodal",
                "page": page_num,
                "chunking_strategy": "multimodal",
                "image_count": len(chunk["images"]),
                "has_vision_descriptions": len(chunk["image_descriptions"]) > 0
            }
        }


# ============================================================================
# SECTION 4: DIAGRAM-AWARE CHUNKING
# ============================================================================

class DiagramChunker:
    """
    Handle diagrams, charts, and visual elements.
    Uses vision models to describe visual content for text-based retrieval.
    """
    
    def __init__(self, vision_model=None):
        self.vision_model = vision_model
    
    def chunk_pdf_diagrams(self, pdf_path: str) -> List[Dict]:
        """Extract and process diagrams/charts from PDF."""
        
        if not HAS_FITZ:
            return [{"text": "PyMuPDF not available", "metadata": {"error": True}}]
        
        chunks = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc, start=1):
            # Try to identify diagrams (heuristic: large images are likely diagrams)
            images = page.get_images(full=True)
            
            for img_idx, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Determine if this might be a diagram (based on size/type)
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    image_size = width * height
                    
                    # Larger images are more likely to be diagrams/charts
                    is_likely_diagram = image_size > 50000  # 50k pixels threshold
                    
                    if is_likely_diagram:
                        diagram_chunk = self._process_diagram(
                            image_bytes,
                            caption=f"Figure on page {page_num}",
                            page_num=page_num
                        )
                        chunks.append(diagram_chunk)
                
                except Exception as e:
                    print(f"   ⚠️ Could not process diagram: {e}")
        
        doc.close()
        return chunks
    
    def _process_diagram(self, image_bytes: bytes, caption: str = "", page_num: int = 1) -> Dict:
        """Convert a diagram into searchable text."""
        
        # Encode as base64 for storage
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Generate description (placeholder - use real vision model in production)
        description = self._describe_diagram(image_bytes) if self.vision_model else self._fallback_description()
        
        # Create searchable text
        searchable_text = f"""
[Diagram/Caption: {caption}]
[Description: {description}]
[Type: visual_content]
"""
        
        return {
            "text": searchable_text.strip(),  # This gets embedded for search
            "raw_table": None,
            "image_data": image_b64[:200] + "...",  # Truncated for display
            "metadata": {
                "type": "diagram",
                "page": page_num,
                "caption": caption,
                "has_image": True,
                "chunking_strategy": "diagram_aware",
                "description": description[:200]
            }
        }
    
    def _describe_diagram(self, image_bytes: bytes) -> str:
        """Use vision model to describe diagram."""
        # Placeholder - replace with actual vision model call
        return "Chart or diagram showing data relationships and trends"
    
    def _fallback_description(self) -> str:
        """Fallback when no vision model available."""
        return "Visual diagram or chart. For detailed analysis, please examine the original document."


# ============================================================================
# SECTION 5: MAIN DEMO RUNNER
# ============================================================================

class MultimodalChunkingDemo:
    """
    Main class to run all chunking strategies on a PDF and compare results.
    """
    
    def __init__(self):
        self.results = {}
    
    def run_demo(self, pdf_path: str):
        """Run all chunking strategies on a single PDF."""
        
        print("\n" + "="*70)
        print("🎨 MULTIMODAL CHUNKING DEMO")
        print("="*70)
        print(f"📄 Input PDF: {pdf_path}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"\n❌ Error: File '{pdf_path}' not found!")
            print("\n📝 Usage: python multimodal_chunker.py path/to/your.pdf")
            return None
        
        # Strategy 1: Table-Aware Chunking
        print("\n" + "📊" * 35)
        print("STRATEGY 1: TABLE-AWARE CHUNKING")
        print("📊" * 35)
        print("   Focus: Preserving table structure and relationships")
        
        table_chunker = TableAwareChunker(max_chunk_size=500)
        table_chunks = table_chunker.chunk_pdf_with_tables(pdf_path)
        self._display_results("Table-Aware", table_chunks)
        
        # Strategy 2: Multimodal Chunking (Text + Images)
        print("\n" + "🖼️" * 35)
        print("STRATEGY 2: MULTIMODAL CHUNKING (Text + Images)")
        print("🖼️" * 35)
        print("   Focus: Keeping images with their surrounding text")
        
        multimodal_chunker = MultimodalChunker(max_chunk_size=500)
        multimodal_chunks = multimodal_chunker.chunk_pdf_multimodal(pdf_path)
        self._display_results("Multimodal", multimodal_chunks)
        
        # Strategy 3: Diagram-Aware Chunking
        print("\n" + "📈" * 35)
        print("STRATEGY 3: DIAGRAM-AWARE CHUNKING")
        print("📈" * 35)
        print("   Focus: Converting visual content to searchable text")
        
        diagram_chunker = DiagramChunker()
        diagram_chunks = diagram_chunker.chunk_pdf_diagrams(pdf_path)
        self._display_results("Diagram-Aware", diagram_chunks)
        
        # Comparison Summary
        self._print_comparison_summary({
            "Table-Aware": len(table_chunks),
            "Multimodal": len(multimodal_chunks),
            "Diagram-Aware": len(diagram_chunks)
        })
        
        # Save results to file
        self._save_results(pdf_path, {
            "table_aware": table_chunks,
            "multimodal": multimodal_chunks,
            "diagram_aware": diagram_chunks
        })
        
        return {
            "table_aware": table_chunks,
            "multimodal": multimodal_chunks,
            "diagram_aware": diagram_chunks
        }
    
    def _display_results(self, strategy_name: str, chunks: List[Dict]):
        """Display formatted results for a strategy."""
        
        if not chunks:
            print("   ⚠️ No chunks generated")
            return
        
        # Count by type
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk.get("metadata", {}).get("type", "unknown")
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        print(f"\n   📦 Total chunks: {len(chunks)}")
        print(f"   📊 Chunk types: {json.dumps(type_counts, indent=2)}")
        
        # Show first 3 chunks
        print("\n   📄 First 3 chunks preview:")
        for i, chunk in enumerate(chunks[:3]):
            chunk_type = chunk.get("metadata", {}).get("type", "unknown")
            text_preview = chunk["text"][:150].replace('\n', ' ')
            print(f"\n   [{i}] Type: {chunk_type}")
            print(f"       Preview: {text_preview}...")
            
            # Show additional info for tables
            if chunk.get("raw_table"):
                print(f"       📋 Has raw table: Yes ({len(chunk['raw_table'])} chars)")
            
            # Show image info
            if chunk.get("has_images") or chunk.get("metadata", {}).get("image_count", 0) > 0:
                img_count = chunk.get("metadata", {}).get("image_count", 0)
                print(f"       🖼️ Contains images: {img_count}")
        
        # Show sample query test
        self._test_query_on_chunks(chunks, strategy_name)
    
    def _test_query_on_chunks(self, chunks: List[Dict], strategy_name: str):
        """Test a sample query on the chunks."""
        
        test_queries = [
            ("table", "revenue", "growth", "sales"),
            ("number", "percentage", "percent", "amount"),
            ("comparison", "vs", "versus", "compare")
        ]
        
        # Simple keyword matching test
        for category, *keywords in test_queries:
            found = False
            for chunk in chunks:
                text_lower = chunk["text"].lower()
                if any(kw in text_lower for kw in keywords):
                    found = True
                    break
            
            if found:
                print(f"\n   🔍 Query '{category} data': ✅ Relevant chunks found")
                break
        else:
            print(f"\n   🔍 Query test: ⚠️ Limited keyword matches (document may be text-light)")
    
    def _print_comparison_summary(self, chunk_counts: Dict):
        """Print a comparison summary of all strategies."""
        
        print("\n" + "="*70)
        print("📊 COMPARISON SUMMARY")
        print("="*70)
        
        print("\n| Strategy | Chunks | Best For |")
        print("|----------|--------|----------|")
        
        descriptions = {
            "Table-Aware": "Documents with data tables, financial reports",
            "Multimodal": "Documents with inline images and captions",
            "Diagram-Aware": "Technical docs with charts and diagrams"
        }
        
        for strategy, count in chunk_counts.items():
            print(f"| {strategy:10} | {count:6} | {descriptions.get(strategy, 'General use')} |")
        
        # Recommendation
        print("\n💡 RECOMMENDATION:")
        if chunk_counts.get("Table-Aware", 0) > 0:
            print("   ✅ Use TABLE-AWARE if your document contains structured data")
        if chunk_counts.get("Multimodal", 0) > 0:
            print("   ✅ Use MULTIMODAL if images are contextually important")
        if chunk_counts.get("Diagram-Aware", 0) > 0:
            print("   ✅ Use DIAGRAM-AWARE for visual search capabilities")
        
        if all(v == 0 for v in chunk_counts.values()):
            print("   ⚠️ No specialized chunks found. Document may be text-only.")
            print("   💡 Use standard RecursiveCharacterTextSplitter instead.")
    
    def _save_results(self, pdf_path: str, results: Dict):
        """Save results to JSON file for later analysis."""
        
        output_dir = "chunking_results"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_{timestamp}_results.json")
        
        # Prepare serializable results
        serializable_results = {}
        for strategy, chunks in results.items():
            serializable_results[strategy] = []
            for chunk in chunks:
                # Create a copy without non-serializable data
                chunk_copy = {
                    "text": chunk.get("text", "")[:1000],  # Limit text length
                    "metadata": chunk.get("metadata", {})
                }
                if chunk.get("raw_table"):
                    chunk_copy["raw_table"] = chunk["raw_table"][:2000]
                if chunk.get("has_images"):
                    chunk_copy["has_images"] = chunk["has_images"]
                if chunk.get("image_count"):
                    chunk_copy["image_count"] = chunk["image_count"]
                
                serializable_results[strategy].append(chunk_copy)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")


# ============================================================================
# SECTION 6: CREATE SAMPLE PDF FOR TESTING
# ============================================================================

def create_sample_pdf(output_path: str = "sample_multimodal_doc.pdf"):
    """Create a sample PDF with text, tables, and images for testing."""
    
    if not HAS_FITZ:
        print("⚠️ Cannot create sample PDF: PyMuPDF not installed")
        return None
    
    doc = fitz.open()
    
    # Page 1: Text and Table
    page1 = doc.new_page()
    y = 50
    
    # Title
    page1.insert_text((50, y), "Q3 2024 Financial Report", fontsize=16)
    y += 40
    
    # Text
    text = "This report summarizes the company's performance in Q3 2024. "
    text += "All departments showed positive growth compared to the previous quarter."
    page1.insert_text((50, y), text, fontsize=11)
    y += 40
    
    # Table data
    table_data = [
        ["Department", "Q2 Revenue", "Q3 Revenue", "Growth"],
        ["Sales", "$1.2M", "$1.5M", "+25%"],
        ["Marketing", "$800K", "$950K", "+19%"],
        ["Engineering", "$500K", "$600K", "+20%"],
        ["Operations", "$400K", "$480K", "+20%"]
    ]
    
    # Draw table
    cell_height = 30
    col_widths = [80, 80, 80, 60]
    x_start = 50
    
    for row_idx, row in enumerate(table_data):
        y_pos = y + (row_idx * cell_height)
        for col_idx, cell in enumerate(row):
            x_pos = x_start + sum(col_widths[:col_idx])
            rect = fitz.Rect(x_pos, y_pos, x_pos + col_widths[col_idx], y_pos + cell_height)
            page1.draw_rect(rect)
            page1.insert_text((x_pos + 5, y_pos + 20), str(cell), fontsize=9)
    
    y += len(table_data) * cell_height + 20
    
    # More text after table
    page1.insert_text((50, y), "As shown above, the Sales department led growth with a 25% increase.", fontsize=11)
    
    # Page 2: More text and image placeholder
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Revenue Growth Trends", fontsize=16)
    page2.insert_text((50, 90), "The chart below illustrates the upward trend across all departments.", fontsize=11)
    
    # Draw a simple bar chart (as diagram)
    chart_y = 130
    page2.draw_rect(fitz.Rect(50, chart_y, 550, chart_y + 200))
    page2.insert_text((60, chart_y + 20), "[Bar Chart Showing Growth %]", fontsize=10)
    page2.insert_text((60, chart_y + 40), "Sales: ████████████████████ 25%", fontsize=9)
    page2.insert_text((60, chart_y + 60), "Marketing: ████████████████ 19%", fontsize=9)
    page2.insert_text((60, chart_y + 80), "Engineering: █████████████████ 20%", fontsize=9)
    page2.insert_text((60, chart_y + 100), "Operations: █████████████████ 20%", fontsize=9)
    
    doc.save(output_path)
    doc.close()
    
    print(f"✅ Created sample PDF: {output_path}")
    return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    MULTIMODAL CHUNKING DEMO                          ║
║  Demonstrates: Table-Aware | Multimodal | Diagram-Aware Chunking     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    if not HAS_FITZ:
        print("❌ PyMuPDF is required. Install with: pip install PyMuPDF")
        sys.exit(1)
    
    # Get PDF path from command line or create sample
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            print(f"\nCreating sample PDF for testing...")
            pdf_path = create_sample_pdf()
            if not pdf_path:
                sys.exit(1)
    else:
        print("📝 No PDF provided. Creating sample PDF for demonstration...")
        pdf_path = create_sample_pdf()
        if not pdf_path:
            print("\n❌ Could not create sample PDF. Please provide a PDF:")
            print("   Usage: python multimodal_chunker.py path/to/your.pdf")
            sys.exit(1)
    
    # Run the demo
    demo = MultimodalChunkingDemo()
    results = demo.run_demo(pdf_path)
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\n📚 KEY TAKEAWAYS:")
    print("   1. TABLE-AWARE: Best for documents with structured data")
    print("   2. MULTIMODAL: Preserves image-text relationships")
    print("   3. DIAGRAM-AWARE: Enables search over visual content")
    print("\n💡 For production use:")
    print("   - Combine strategies based on document type")
    print("   - Use vision models (GPT-4V, Llama 3.2) for accurate diagram descriptions")
    print("   - Store raw tables for LLM context, linearized for embedding")


if __name__ == "__main__":
    main()