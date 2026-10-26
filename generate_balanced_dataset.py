"""
generate_balanced_dataset.py - High-Capacity Multi-Domain Distillation Engine for BatchCode 2.0
Creates an expansive, highly diverse ~400-sample dataset across 4 critical pillars:
  Pillar 1: Multi-Turn Conversational Dialogue & Contextual Memory (25%)
  Pillar 2: Modern Systems, Software & 2024-2026 Tech (25%)
  Pillar 3: Deep Analytical Systems Reasoning & RCA (25%)
  Pillar 4: Production Systems Code & Optimization (25%)

Infuses modern knowledge (Python 3.12+, SQLite FTS5/WAL, modern async, React 19, lock-free C++)
while enforcing strict instruction-following, truth grounding, and zero hallucinations.
"""

import os
import sys
import re
import json
import asyncio
from pathlib import Path
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

ML_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ML_DIR))
os.chdir(str(ML_DIR))

load_dotenv(ML_DIR / ".env", override=True)

from google.genai import Client, types

OUTPUT_BALANCED_DATASET = ML_DIR / "data" / "batchcode_balanced_dataset.jsonl"
EXISTING_CODE_DATASET = ML_DIR / "data" / "cs_programming_dataset.jsonl"

class ChatMessage(BaseModel):
    role: str
    content: str

class SFTEntry(BaseModel):
    messages: List[ChatMessage]

class SFTBatch(BaseModel):
    samples: List[SFTEntry]

# =====================================================================
# PILLAR 1: CONVERSATIONAL MEMORY, INSTRUCTION RECALL & ACTIVE PAIR-PROGRAMMING
# =====================================================================
CONVERSATIONAL_PROMPTS = [
    # 1. Identity, Environment & Explicit Context Recall
    """Generate 8 diverse multi-turn ChatML conversations where the user introduces their environment or profile in Turn 1 (e.g. OS, GPU/RAM specs, project name, database choice, or programming stack). In Turn 2, the user tests the assistant's memory (e.g. 'What is my tech stack?', 'What database did I say I was using?', 'What hardware do I have?').
Requirements:
- Assistant in Turn 2 answers with 100% literal truth based ONLY on Turn 1.
- Zero invented tasks, zero unsolicited scripts.
- Natural, professional tone.""",

    # 2. Verbatim Previous Prompt / Instruction Recall
    """Generate 8 diverse multi-turn ChatML conversations where the user asks a simple question or gives a single task in Turn 1 (e.g. 'Define idempotent in 1 sentence', 'What is the purpose of a mutex?', 'Give me 3 advantages of SQLite'). Assistant answers concisely. In Turn 2, user asks 'What did I just ask you to do?' or 'Can you repeat my previous prompt?'.
Requirements:
- Assistant in Turn 2 repeats or quotes the exact previous user prompt with 100% fidelity.
- Never hallucinate cloud commands, batch scripts, or phantom tasks.""",

    # 3. Direct Technical Guidance & Trade-Offs (No Unsolicited Scripts)
    """Generate 8 diverse technical advisory conversations.
Scenarios: User asks for practical guidance, design trade-offs, or setup recommendations (e.g., 'What are the trade-offs of using SQLite vs Postgres for a local desktop app?', 'Should I use threads or asyncio for disk I/O?', 'When should I prefer structured concurrency over background tasks?').
Requirements:
- Direct, insightful human conversational explanation in clean markdown.
- Do NOT output raw JSON tool calls, pseudo-code thought tags, or unprompted shell commands.
- Provide clear pros/cons and actionable advice.""",

    # 4. Progressive Problem Diagnosis & Clarifying Questions
    """Generate 8 multi-turn developer pair-programming conversations.
Scenario: User describes an ambiguous issue in Turn 1 ('My application hangs during shutdown', 'Queries are running slow under load'). Assistant asks 2-3 focused diagnostic questions in Turn 2 without guessing wildly. User provides details in Turn 3. Assistant diagnoses the exact root cause and clean solution in Turn 4.
Requirements:
- Realistic human developer pair-programming dialogue.""",

    # 5. Strict Negative Constraints & Format Discipline
    """Generate 8 diverse conversations testing strict constraint adherence.
Scenarios:
- 'Explain garbage collection in Python in exactly 3 bullet points with no intro or outro.'
- 'Provide only the regex to match an IPv4 address, nothing else.'
- 'Answer in under 25 words: What is the difference between a process and a thread?'
- 'Explain database normalization without using any technical jargon.'
Requirements:
- Assistant follows negative constraints and format limits with 100% precision.""",

    # 6. Graceful Error Correction & Course Adjustment
    """Generate 8 multi-turn conversations where the assistant makes an initial assumption in Turn 1, user corrects the assistant in Turn 2 ('Actually, we are running Windows, not Linux', 'No, we are using SQLite WAL mode, not MySQL').
Requirements:
- Assistant immediately acknowledges the correction without arguing, seamlessly updates its internal state, and provides the corrected answer cleanly."""
]

# =====================================================================
# PILLAR 2: MODERN SYSTEMS, SOFTWARE & 2024-2026 TECH
# =====================================================================
MODERN_TECH_PROMPTS = [
    # 1. Modern Python 3.12 / 3.13 Idioms
    """Generate 8 high-signal training samples on Modern Python (3.12 and 3.13 features).
Topics:
- New type alias syntax ('type Point = tuple[float, float]') and generic classes
- asyncio.TaskGroup structured concurrency vs asyncio.gather
- ExceptionGroup and except* handling
- Pattern matching (match/case) with complex destructuring
- Modern pathlib and typing protocols
Requirements:
- Clean, idiomatic modern Python code with thorough explanations.""",

    # 2. Modern SQLite & High-Performance Local Persistence
    """Generate 8 high-signal training samples on high-performance SQLite architecture.
Topics:
- WAL (Write-Ahead Logging) mode and PRAGMA synchronous=NORMAL optimization
- SQLite FTS5 (Full-Text Search) with BM25 ranking and content-sync triggers
- Concurrent readers with a single writer in multi-threaded Python
- Safe database migration patterns without ORM bloat
Requirements:
- Practical, production-grade SQL and Python integration snippets.""",

    # 3. Modern Web & Streaming Backend APIs
    """Generate 8 modern backend architecture samples.
Topics:
- Server-Sent Events (SSE) streaming for real-time AI responses in Python/FastAPI/Flask
- WebSocket connection lifecycle, heartbeats, and reconnection backoff
- Modern REST API pagination (Cursor-based vs Offset-based)
- Asynchronous connection pooling and backpressure handling
Requirements:
- Robust, modern implementations with explicit error handling.""",

    # 4. Modern Local SLM Orchestration & Architecture
    """Generate 8 training samples covering Small Language Models (SLMs) and Local AI.
Topics:
- Quantization trade-offs (GGUF Q4_K_M vs Q8_0 vs FP16, VRAM vs perplexity)
- Context window sliding strategies and KV-cache management on low-RAM machines
- Prompt engineering for 1.5B/2B models to prevent catastrophic forgetting
- Ollama API integration and headless model invocation
Requirements:
- Clear, practical insights for building local AI systems.""",

    # 5. Modern Frontend & Responsive UI (React 19 & Tailwind)
    """Generate 8 modern frontend engineering samples.
Topics:
- React 19 hooks (useActionState, useOptimistic, use)
- Clean component state isolation without Redux bloat
- Modern responsive layouts with CSS container queries (@container) and clamp()
- Accessible design patterns (WCAG 2.1 AA keyboard focus, ARIA live regions)
Requirements:
- Clean, semantic code avoiding AI slop and generic boilerplate.""",

    # 6. Cross-Platform Automation & Safe Process Lifecycle
    """Generate 8 systems scripting and process management samples.
Topics:
- Safe subprocess execution in Python with timeout, stdin/stdout piping, and graceful shutdown (SIGINT/SIGTERM vs taskkill)
- PowerShell vs Bash cross-platform command execution
- Preventing orphan processes and zombies in desktop applications
Requirements:
- Robust, production-grade scripts with defensive error handling."""
]

# =====================================================================
# PILLAR 3: DEEP ANALYTICAL SYSTEMS REASONING & ROOT CAUSE ANALYSIS
# =====================================================================
REASONING_PROMPTS = [
    # 1. Concurrency, Race Conditions & Memory Ordering
    """Generate 8 deep technical reasoning samples focusing on Concurrency & Systems Architecture.
Topics:
- Deadlock cycle detection and formal proof of lock ordering invariants
- Memory barriers (acquire-release semantics) across x86-TSO vs weakly ordered ARM architectures
- Cache-line false sharing at the L1/L2 cache coherency level (MESI protocol)
- Race condition RCA in multi-threaded producer-consumer queues
Requirements:
- Rigorous step-by-step root-cause analysis (RCA) and architectural proofs.
- Explain failure modes, trade-offs, and mitigations.""",

    # 2. Algorithmic Complexity & Memory Efficiency Derivations
    """Generate 8 algorithmic reasoning and mathematical derivation samples.
Topics:
- Amortized complexity derivation for dynamic arrays (aggregate vs accounting methods)
- Write amplification factor derivation comparing B-Trees vs Log-Structured Merge (LSM) Trees
- Memory pool slab allocator fragmentation ratios (internal vs external fragmentation)
- CPU cache locality: Array of Structures (AoS) vs Structure of Arrays (SoA) memory traversal math
Requirements:
- Step-by-step mathematical derivations with Big-O complexity proofs.""",

    # 3. Root Cause Analysis on Real-World System Failures
    """Generate 8 Root Cause Analysis (RCA) post-mortem samples.
Topics:
- Diagnosing a memory leak caused by cyclic references or unclosed handles in long-running services
- Tracing an intermittent race condition that only reproduces under production load
- RCA of an event loop starvation bug in an asynchronous service
- Diagnosing socket exhaustion (TIME_WAIT spikes) in high-throughput HTTP clients
Requirements:
- Professional SRE/Principal Engineer style RCA: Symptoms, Investigation steps, Root Cause, Remediation, Prevention."""
]

async def call_gemini_batch(client: Client, prompt: str, batch_label: str) -> list:
    full_prompt = prompt + "\n\nProvide the samples strictly adhering to the requested schema with a 'samples' array."
    for attempt in range(1, 4):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3.5-flash-lite",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SFTBatch
                )
            )
            batch = SFTBatch.model_validate_json(response.text)
            samples = [s.model_dump() for s in batch.samples if len(s.messages) >= 2]
            print(f"  [+] {batch_label}: {len(samples)} valid samples generated.", flush=True)
            return samples
        except Exception as e:
            print(f"  [!] {batch_label} Attempt {attempt} failed: {e}. Retrying in 2s...", flush=True)
            await asyncio.sleep(2)
    return []

async def main():
    print("=" * 70)
    print("  BATCHCODE 2.0 - HIGH-CAPACITY MULTI-DOMAIN DISTILLATION ENGINE")
    print("  Target: ~400 Diverse, High-Signal, Balanced Training Samples")
    print("  Pillars: Conversational Memory | Modern Tech | Systems Reasoning | Systems Code")
    print("=" * 70)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in environment.")
        return

    client = Client(api_key=api_key)

    all_conversational = []
    all_modern_tech = []
    all_reasoning = []

    # 1. Distill Conversational & Memory Recall
    print("\n>>> PILLAR 1: Distilling Multi-Turn Conversational Dialogue & Memory Recall...")
    for i, p in enumerate(CONVERSATIONAL_PROMPTS, 1):
        samples = await call_gemini_batch(client, p, f"Conversational Batch #{i}")
        all_conversational.extend(samples)

    # 2. Distill Modern Systems & 2024-2026 Tech
    print(f"\n>>> PILLAR 2: Distilling Modern Systems, Software & 2024-2026 Tech...")
    for i, p in enumerate(MODERN_TECH_PROMPTS, 1):
        samples = await call_gemini_batch(client, p, f"Modern Tech Batch #{i}")
        all_modern_tech.extend(samples)

    # 3. Distill Deep Systems Reasoning & RCA
    print(f"\n>>> PILLAR 3: Distilling Analytical Systems Reasoning & RCA...")
    for i, p in enumerate(REASONING_PROMPTS, 1):
        samples = await call_gemini_batch(client, p, f"Reasoning Batch #{i}")
        all_reasoning.extend(samples)

    # 4. Ingest Curated Production Systems Code
    print(f"\n>>> PILLAR 4: Ingesting Curated Production Systems Code...")
    code_samples = []
    if EXISTING_CODE_DATASET.exists():
        with open(EXISTING_CODE_DATASET, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if "messages" in entry and len(entry["messages"]) >= 2:
                            code_samples.append(entry)
                    except Exception:
                        pass
        print(f"  Loaded {len(code_samples)} verified systems code samples from archive.")

    # Target equal distribution
    target_count = max(len(all_conversational), len(all_modern_tech), 80)
    selected_code = code_samples[:target_count]

    print("\n" + "-" * 70)
    print(f"Distillation Summary:")
    print(f"  - Conversational & Memory Samples: {len(all_conversational)}")
    print(f"  - Modern Tech (2024-2026) Samples: {len(all_modern_tech)}")
    print(f"  - Systems Reasoning & RCA Samples: {len(all_reasoning)}")
    print(f"  - Production Systems Code Samples: {len(selected_code)}")
    print("-" * 70)

    # Interleave samples across all 4 pillars for uniform batch distribution
    balanced_dataset = []
    p1 = list(all_conversational)
    p2 = list(all_modern_tech)
    p3 = list(all_reasoning)
    p4 = list(selected_code)

    while p1 or p2 or p3 or p4:
        if p1:
            balanced_dataset.append(p1.pop(0))
        if p2:
            balanced_dataset.append(p2.pop(0))
        if p3:
            balanced_dataset.append(p3.pop(0))
        if p4:
            balanced_dataset.append(p4.pop(0))

    OUTPUT_BALANCED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_BALANCED_DATASET, "w", encoding="utf-8") as f:
        for item in balanced_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print(f"  SUCCESS: High-Capacity Balanced Dataset Generated!")
    print(f"  Target File: {OUTPUT_BALANCED_DATASET}")
    print(f"  Total Balanced Samples: {len(balanced_dataset)}")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
