from typing import List, Dict, Any

class ContextualChunker:
    """Implements Anthropic's Contextual Retrieval chunking pattern.
    Prepends document-level and section-level context to each chunk so
    retrievers do not lose critical financial reference points (company, year, section).
    """

    def __init__(self, doc_title: str = "Apple Inc. Form 10-K (FY2024)"):
        self.doc_title = doc_title

    def chunk_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        chunk_id = 0

        for block in blocks:
            content = block["content"]
            b_type = block["type"]
            breadcrumb = block.get("breadcrumb", [])
            
            section_path = " > ".join(breadcrumb) if breadcrumb else "General"
            
            # Contextual prefix (Anthropic Contextual Retrieval pattern)
            context_prefix = (
                f"[Document: {self.doc_title} | Section: {section_path} | Type: {b_type.upper()}]"
            )
            
            # Keep tables atomic to preserve tabular numerical integrity
            if b_type == "table":
                contextual_text = f"{context_prefix}\n\nTable Data:\n{content}"
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "raw_content": content,
                    "contextual_content": contextual_text,
                    "type": "table",
                    "section": section_path,
                    "doc_title": self.doc_title
                })
                chunk_id += 1
            else:
                # Text block - if small enough, keep as single chunk
                words = content.split()
                if len(words) <= 300:
                    contextual_text = f"{context_prefix}\n\n{content}"
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_id}",
                        "raw_content": content,
                        "contextual_content": contextual_text,
                        "type": "text",
                        "section": section_path,
                        "doc_title": self.doc_title
                    })
                    chunk_id += 1
                else:
                    # Split into overlapping windows of ~250 words
                    window_size = 250
                    overlap = 40
                    step = window_size - overlap
                    for i in range(0, len(words), step):
                        sub_content = " ".join(words[i:i + window_size])
                        contextual_text = f"{context_prefix}\n\n{sub_content}"
                        chunks.append({
                            "chunk_id": f"chunk_{chunk_id}",
                            "raw_content": sub_content,
                            "contextual_content": contextual_text,
                            "type": "text",
                            "section": section_path,
                            "doc_title": self.doc_title
                        })
                        chunk_id += 1

        return chunks
