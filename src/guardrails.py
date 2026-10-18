import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

class Citation(BaseModel):
    chunk_id: str = Field(description="The unique identifier of the retrieved source chunk")
    section: str = Field(description="The section or table breadcrumb where the fact was found")
    quote_snippet: str = Field(description="Exact verbatim excerpt from context supporting the claim")

class VerifiedMetric(BaseModel):
    metric_name: str = Field(description="Name of the financial metric, e.g. Services Revenue FY24")
    value: str = Field(description="Reported value, e.g. $96,169 million or 46.2%")
    is_grounded: bool = Field(description="True if exact value is confirmed present in context")

class GroundedFinancialResponse(BaseModel):
    """Pydantic Schema enforcing strict factual grounding and structured citations."""
    executive_summary: str = Field(description="Concise, factual executive summary of the answer")
    detailed_findings: str = Field(description="Detailed breakdown with references to tables or sections")
    citations: List[Citation] = Field(default_factory=list, description="List of source citations")
    verified_metrics: List[VerifiedMetric] = Field(default_factory=list, description="Extracted numerical figures")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Model confidence based on context sufficiency")
    guardrail_status: str = Field(default="PASSED", description="Guardrail audit status: PASSED or FLAGGED")

class AIGuardrails:
    """Enterprise AI Guardrails Suite:
    1. Prompt Injection & Adversarial Attack Defense
    2. Numerical Hallucination Verification
    3. Context Sufficiency Assessment
    """

    ADVERSARIAL_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?prior\s+guidelines",
        r"system\s+prompt\s+override",
        r"jailbreak",
        r"drop\s+table",
        r"delete\s+from",
        r"reveal\s+(your\s+)?system\s+instructions"
    ]

    @classmethod
    def sanitize_input(cls, user_query: str) -> Tuple[bool, str]:
        """Validates input against known prompt injection and jailbreak vectors.
        Returns: (is_safe: bool, sanitized_or_reason: str)
        """
        for pattern in cls.ADVERSARIAL_PATTERNS:
            if re.search(pattern, user_query, re.IGNORECASE):
                return False, f"Query blocked by AI Input Guardrail: Potential prompt injection detected matching rule '{pattern}'."
        
        # Strip excessive whitespace / hidden escape chars
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", user_query).strip()
        return True, cleaned

    @classmethod
    def verify_numerical_hallucinations(
        cls,
        answer_text: str,
        retrieved_contexts: List[str]
    ) -> List[VerifiedMetric]:
        """Extracts numerical metrics from the answer and verifies if they strictly
        exist in the retrieved context chunks.
        """
        combined_context = " ".join(retrieved_contexts).replace(",", "")
        
        # Regex to capture currency figures and percentages: e.g. $201,183M, 46.2%, $96.2 billion
        pattern = r"(\$[\d,.]+\s*(?:billion|million|B|M)?|\d+\.?\d*%)"
        matches = re.findall(pattern, answer_text, re.IGNORECASE)
        
        verified_list: List[VerifiedMetric] = []
        for match in set(matches):
            raw_match = match.strip()
            # Normalize for matching
            num_clean = re.sub(r"[$,]", "", raw_match).split()[0]
            is_present = num_clean in combined_context
            
            verified_list.append(VerifiedMetric(
                metric_name=f"Reported value '{raw_match}'",
                value=raw_match,
                is_grounded=is_present
            ))
            
        return verified_list
