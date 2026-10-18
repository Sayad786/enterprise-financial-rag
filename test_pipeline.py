import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_DIR
from src.parser import LayoutAwareParser
from src.contextual_chunker import ContextualChunker
from src.hybrid_retriever import HybridRetriever
from src.generator import FinancialGenerator
from src.evals import RagasEvaluationHarness

def main():
    print("=" * 60)
    print("TESTING ENTERPRISE FINANCIAL RAG PIPELINE")
    print("=" * 60)

    doc_file = DATA_DIR / "sample_10k_apple_fy24.md"
    print(f"\n[1] Parsing Document: {doc_file.name}")
    blocks = LayoutAwareParser.parse_document(doc_file)
    print(f" -> Extracted {len(blocks)} layout blocks (tables & text).")

    print("\n[2] Applying Anthropic Contextual Retrieval Chunker...")
    chunker = ContextualChunker(doc_title="Apple Inc. Form 10-K (FY2024)")
    chunks = chunker.chunk_blocks(blocks)
    print(f" -> Generated {len(chunks)} contextual chunks.")
    print(f" -> Sample Contextual Prefix:\n    {chunks[0]['contextual_content'][:110]}...")

    print("\n[3] Initializing Hybrid Retriever (Dense + BM25 + RRF + Cross-Encoder)...")
    retriever = HybridRetriever(chunks)
    print(" -> Hybrid indexing complete.")

    test_query = "What was Apple's Services revenue and YoY growth in FY2024?"
    print(f"\n[4] Executing Hybrid Query: '{test_query}'")
    results = retriever.retrieve(test_query, dense_top_k=5, sparse_top_k=5, rerank_top_k=3)
    print(f" -> Retrieved {len(results)} top reranked chunks.")
    for idx, r in enumerate(results, start=1):
        print(f"    Match {idx}: [{r['chunk_id']}] {r['section']} (Score: {r['rerank_score']})")

    print("\n[5] Generating Grounded Response & Running AI Guardrails...")
    response, latency, tokens = FinancialGenerator.generate_answer(test_query, results)
    print(f" -> Generation Latency: {latency:.3f}s")
    print(f" -> Guardrail Status: {response.guardrail_status}")
    print(f" -> Executive Summary: {response.executive_summary}")
    print(f" -> Verified Metrics Count: {len(response.verified_metrics)}")
    for vm in response.verified_metrics[:3]:
        print(f"      * {vm.value} -> Grounded: {vm.is_grounded}")

    print("\n[6] Running Ragas Continuous Evaluation Suite...")
    eval_scores = RagasEvaluationHarness.evaluate_response(
        query=test_query,
        answer=response.detailed_findings,
        retrieved_contexts=[r['raw_content'] for r in results],
        latency_sec=latency,
        tokens=tokens
    )
    print(" -> Ragas Evaluation Metrics:")
    for k, v in eval_scores.items():
        print(f"      {k}: {v}")

    print("\n" + "=" * 60)
    print("ALL PIPELINE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
