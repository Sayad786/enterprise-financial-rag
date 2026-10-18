import re
from typing import List, Dict, Any
from pathlib import Path

class LayoutAwareParser:
    """Parses financial reports (Markdown / Text) into structured blocks,
    preserving tables and hierarchical section breadcrumbs.
    """
    
    @staticmethod
    def is_table_row(line: str) -> bool:
        line = line.strip()
        return line.startswith("|") and line.endswith("|")
    
    @classmethod
    def parse_document(cls, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        blocks: List[Dict[str, Any]] = []
        
        current_breadcrumb: List[str] = []
        buffer_lines: List[str] = []
        in_table = False
        
        for line in lines:
            trimmed = line.strip()
            
            # Header Detection (H1, H2, H3)
            header_match = re.match(r"^(#{1,3})\s+(.*)$", trimmed)
            if header_match:
                # Flush existing buffer
                if buffer_lines:
                    text_block = "\n".join(buffer_lines).strip()
                    if text_block:
                        blocks.append({
                            "type": "table" if in_table else "text",
                            "content": text_block,
                            "breadcrumb": list(current_breadcrumb)
                        })
                    buffer_lines = []
                    in_table = False
                
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Maintain breadcrumb hierarchy
                if level == 1:
                    current_breadcrumb = [title]
                elif level == 2:
                    current_breadcrumb = current_breadcrumb[:1] + [title]
                elif level == 3:
                    current_breadcrumb = current_breadcrumb[:2] + [title]
                continue
            
            # Table Row Detection
            if cls.is_table_row(trimmed):
                if not in_table:
                    # Flush previous text before table starts
                    if buffer_lines:
                        text_block = "\n".join(buffer_lines).strip()
                        if text_block:
                            blocks.append({
                                "type": "text",
                                "content": text_block,
                                "breadcrumb": list(current_breadcrumb)
                            })
                        buffer_lines = []
                    in_table = True
                buffer_lines.append(line)
            else:
                if in_table:
                    # Table just finished, flush it
                    table_block = "\n".join(buffer_lines).strip()
                    if table_block:
                        blocks.append({
                            "type": "table",
                            "content": table_block,
                            "breadcrumb": list(current_breadcrumb)
                        })
                    buffer_lines = []
                    in_table = False
                
                # Normal paragraph line
                if trimmed:
                    buffer_lines.append(line)
                elif buffer_lines:
                    # Empty line acts as paragraph boundary
                    text_block = "\n".join(buffer_lines).strip()
                    if text_block:
                        blocks.append({
                            "type": "text",
                            "content": text_block,
                            "breadcrumb": list(current_breadcrumb)
                        })
                    buffer_lines = []

        # Flush any remaining buffer
        if buffer_lines:
            text_block = "\n".join(buffer_lines).strip()
            if text_block:
                blocks.append({
                    "type": "table" if in_table else "text",
                    "content": text_block,
                    "breadcrumb": list(current_breadcrumb)
                })

        return blocks
