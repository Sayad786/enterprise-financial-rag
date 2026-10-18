import json
import re
import time
from typing import List, Dict, Any, Tuple

from src.config import Config
from src.guardrails import AIGuardrails, GroundedFinancialResponse, Citation, VerifiedMetric

SYSTEM_PROMPT = """You are a Principal Financial Analyst and Regulatory AI Auditor.
Your objective is to answer questions strictly using the provided context from SEC 10-K filings.

RULES:
1. Grounding: Answer ONLY based on the provided context. If the information is not present, explicitly state that it is unavailable.
2. Numbers: State exact financial metrics (revenue, percentages, dates) precisely as reported.
3. Citations: Cite the exact Section and Table from which each number or insight is derived.
4. Output Schema: You MUST respond in valid JSON conforming to the following structure:
{
  "executive_summary": "1-2 sentence high-level summary",
  "detailed_findings": "Detailed breakdown with bullet points and table references",
  "citations": [
    {"chunk_id": "chunk_0", "section": "Item 8 Consolidated Statements", "quote_snippet": "exact quote"}
  ],
  "confidence_score": 0.95
}
"""

class FinancialGenerator:
    """Orchestrates LLM generation with strict AI Guardrails, Pydantic validation,
    and automatic fallback to deterministic extraction in Zero-Cost Local Mode.
    """

    @classmethod
    def generate_answer(
        cls,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[GroundedFinancialResponse, float, Dict[str, int]]:
        """Generates a grounded financial response.
        Returns: (GroundedFinancialResponse, latency_seconds, token_telemetry)
        """
        start_time = time.time()
        
        # 1. Input Guardrail
        is_safe, message = AIGuardrails.sanitize_input(query)
        if not is_safe:
            latency = time.time() - start_time
            return GroundedFinancialResponse(
                executive_summary="Query Rejected by Safety Guardrail",
                detailed_findings=message,
                confidence_score=0.0,
                guardrail_status="BLOCKED_BY_GUARDRAIL"
            ), latency, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # 2. Format Context Chunks
        context_blocks = []
        for c in retrieved_chunks:
            context_blocks.append(
                f"--- [SOURCE ID: {c['chunk_id']} | SECTION: {c['section']} | RERANK SCORE: {c.get('rerank_score', 'N/A')}] ---\n"
                f"{c['raw_content']}\n"
            )
        formatted_context = "\n\n".join(context_blocks)

        user_prompt = f"USER QUERY:\n{query}\n\nRELEVANT 10-K CONTEXT CHUNKS:\n{formatted_context}"

        # 3. Cloud LLM Calling (Anthropic or OpenAI)
        if Config.is_anthropic_available():
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
                resp = client.messages.create(
                    model=Config.LLM_MODEL,
                    max_tokens=1500,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                raw_text = resp.content[0].text
                parsed_json = cls._extract_json(raw_text)
                latency = time.time() - start_time
                tokens = {
                    "prompt_tokens": resp.usage.input_tokens,
                    "completion_tokens": resp.usage.output_tokens,
                    "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens
                }
                return cls._build_and_verify_response(parsed_json, retrieved_chunks), latency, tokens
            except Exception as e:
                print(f"[Warning] Anthropic LLM call failed, falling back to local deterministic engine: {e}")

        # 4. Zero-Cost Local Deterministic Reasoning Engine
        # Formulates an exact, high-precision grounded response directly from the retrieved chunks
        response, tokens = cls._local_reasoning_engine(query, retrieved_chunks)
        latency = time.time() - start_time
        return response, latency, tokens

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {
            "executive_summary": "Extracted response from context",
            "detailed_findings": text,
            "citations": [],
            "confidence_score": 0.85
        }

    @classmethod
    def _local_reasoning_engine(
        cls,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[GroundedFinancialResponse, Dict[str, int]]:
        """High-precision local extraction for immediate offline execution without API keys."""
        if not chunks:
            return GroundedFinancialResponse(
                executive_summary="No relevant financial documents found.",
                detailed_findings="The retrieval system could not locate sections matching this query.",
                confidence_score=0.0,
                guardrail_status="PASSED"
            ), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        top_chunk = chunks[0]
        citations = [
            Citation(
                chunk_id=c["chunk_id"],
                section=c["section"],
                quote_snippet=c["raw_content"][:160] + "..."
            )
            for c in chunks[:3]
        ]

        # Synthesize executive summary and findings
        summary = (
            f"Based on {top_chunk['doc_title']} ({top_chunk['section']}), "
            f"the analysis identifies key operational metrics relevant to: '{query}'."
        )
        
        findings_lines = [
            f"### Key Findings from {top_chunk['section']}:",
            ""
        ]
        
        # Include relevant content lines or table
        for c in chunks[:2]:
            findings_lines.append(f"**Source: {c['section']} (Relevance Score: {c.get('rerank_score', 0.95)}):**")
            findings_lines.append(c['raw_content'])
            findings_lines.append("")

        detailed_findings = "\n".join(findings_lines)

        # Numerical Hallucination Verification
        context_texts = [c["raw_content"] for c in chunks]
        verified_metrics = AIGuardrails.verify_numerical_hallucinations(detailed_findings, context_texts)

        response = GroundedFinancialResponse(
            executive_summary=summary,
            detailed_findings=detailed_findings,
            citations=citations,
            verified_metrics=verified_metrics,
            confidence_score=float(top_chunk.get("rerank_score", 0.92)),
            guardrail_status="PASSED"
        )
        
        # Token count estimation
        estimated_prompt_tokens = sum(len(c["raw_content"].split()) for c in chunks) + len(query.split()) + 150
        estimated_completion_tokens = len(detailed_findings.split()) + 40
        tokens = {
            "prompt_tokens": estimated_prompt_tokens,
            "completion_tokens": estimated_completion_tokens,
            "total_tokens": estimated_prompt_tokens + estimated_completion_tokens
        }
        return response, tokens

    @classmethod
    def _build_and_verify_response(
        cls,
        parsed_json: Dict[str, Any],
        chunks: List[Dict[str, Any]]
    ) -> GroundedFinancialResponse:
        context_texts = [c["raw_content"] for c in chunks]
        summary = parsed_json.get("executive_summary", "")
        findings = parsed_json.get("detailed_findings", "")
        
        raw_citations = parsed_json.get("citations", [])
        citations = []
        for c in raw_citations:
            citations.append(Citation(
                chunk_id=c.get("chunk_id", "chunk_0"),
                section=c.get("section", "SEC 10-K Filing"),
                quote_snippet=c.get("quote_snippet", "")
            ))
            
        verified_metrics = AIGuardrails.verify_numerical_hallucinations(
            f"{summary} {findings}", context_texts
        )

        return GroundedFinancialResponse(
            executive_summary=summary,
            detailed_findings=findings,
            citations=citations,
            verified_metrics=verified_metrics,
            confidence_score=float(parsed_json.get("confidence_score", 0.95)),
            guardrail_status="PASSED"
        )
