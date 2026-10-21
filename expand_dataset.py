"""
expand_dataset.py - High-Throughput Distillation Engine for 'batchcode'
Generates ~450 new verified, sanitized samples across:
  - 250 Programming & Syntax samples
  - 125 Deep Technical Reasoning & Problem Decomposition samples
  - 75 Tool Use & Agentic Function Calling samples
Total dataset size: ~500 samples
"""

import asyncio
import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" in os.environ:
    del os.environ["GOOGLE_API_KEY"]

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from app.distill_engine import (
    call_gemini_with_backoff,
    validate_sample,
    estimate_tokens,
    sanitize_text,
    commit_to_jsonl
)

DATASET_LOCAL = "data/cs_programming_dataset.jsonl"
DATASET_DOWNLOADS = r"C:\Users\Ryan\Downloads\batchcode_latest\datacs_programming_dataset.jsonl"

PROGRAMMING_TOPICS = [
    ("Modern C++20 Systems & Atomics", "Advanced C++20/C++23 systems programming focusing on atomic memory order (acquire/release/relaxed), lock-free queues, std::atomic<std::shared_ptr>, and memory barriers."),
    ("Linux POSIX Epoll & Non-Blocking I/O", "High-performance network programming in C using epoll_create1, epoll_ctl, non-blocking sockets, edge-triggered I/O, and event loops."),
    ("Custom Memory Allocators in C/C++", "Low-level memory management implementing slab allocators, buddy allocators, memory arena pools, cache-line alignment, and zero fragmentation strategies."),
    ("Rust Safe Systems & Concurrency", "Senior-level Rust implementations using Arc, Mutex, RwLock, crossbeam channels, atomic primitives, and zero-cost abstractions for safe multithreading."),
    ("Go Concurrency & Channel Multiplexing", "High-throughput Go services using worker pools, channel select multiplexing, sync.Pool, sync.WaitGroup, context cancellation, and race condition prevention."),
    ("Python High-Performance & Internals", "Low-level Python using memoryview, buffer protocol, ctypes/C-extensions, custom bytecode inspection, and asyncio event loop internals."),
    ("Database Storage Engines & B+ Trees", "Database storage internals implementing B+ Tree node splitting/merging, Write-Ahead Logging (WAL) append mechanisms, and LRU/Clock buffer pools."),
    ("Algorithmic Graph Theory & Flow", "Production-grade implementations of Dinic's blocking flow, Tarjan's strongly connected components, segment trees with lazy propagation, and Min-Cut algorithms."),
    ("Compiler & Parser Engineering", "Writing recursive descent AST parsers, Pratt precedence expression parsing, lexer state machines, and bytecode VM execution stacks."),
    ("Distributed Consensus & Network Primitives", "Building distributed primitives including Raft leader election state machines, consistent hashing rings with virtual nodes, and vector clocks.")
]

REASONING_TOPICS = [
    ("Concurrency Race Condition & Deadlock RCA", "Detailed step-by-step root-cause analysis and mathematical reasoning identifying deadlock cycles (lock ordering), priority inversions, and ABA race conditions."),
    ("Algorithmic Complexity Proofs & Amortized Bounds", "Rigorous mathematical derivations of computational time and space complexity, master theorem applications, potential method for amortized bounds, and cache miss models."),
    ("Cache Coherence, False Sharing & CPU Architecture", "Deep technical analysis and reasoning about CPU L1/L2/L3 cache hierarchies, cache line bouncing, false sharing, and memory layout optimization (Structure-of-Arrays vs. Array-of-Structures)."),
    ("Distributed Systems Failure Recovery & CAP Tradeoffs", "Systematic reasoning and architectural proof for handling network partitions, split-brain scenarios, quorum math in Raft/Paxos, and consistency tradeoffs under the CAP theorem."),
    ("Kernel Subsystem Memory Corruption RCA", "Analytical breakdown diagnosing kernel panics, stack overflow in deep recursion, page fault invalid address traps, memory leak post-mortems, and heap corruption.")
]

TOOL_USE_TOPICS = [
    ("Shell Command Execution & Terminal Agent", "Agent tool-calling workflows where the user requests a task and the assistant generates structured tool calls like <tool_call>{\"name\": \"run_command\", \"arguments\": {\"command\": \"...\"}}</tool_call>, handles command output, and verifies execution."),
    ("Codebase Search & AST Symbol Navigation", "Tool-use interactions using code search and symbol navigation tools to find class declarations, inspect function usages across files, and report findings in structured format."),
    ("Git Workflow & Diff Resolution", "Tool-calling scenarios for git operations: inspecting diffs, resolving merge conflicts, creating release tags, and checking commit histories via tool calls."),
    ("Database Query & Schema Inspection Tools", "Agent interactions executing SQL queries via database tools, inspecting table schemas, analyzing EXPLAIN plans, and repairing query bottlenecks."),
    ("REST API Client & Service Orchestration", "Tool use constructing HTTP client calls, handling auth headers, parsing JSON responses, and retrying on transient errors via tool definitions.")
]

async def distill_topic_batch(topic_name: str, topic_desc: str, count: int, category: str) -> list:
    print(f"\n[{category}] Generating {count} samples for '{topic_name}'...", flush=True)
    
    trait_map = {
        "Tool Use": "tool_use",
        "Reasoning": "reasoning",
        "Programming": "coding"
    }
    trait = trait_map.get(category, "coding")

    if category == "Tool Use":
        prompt = f"""You are an elite AI Coding Agent capable of structured tool calling and autonomous execution.
Target Topic: {topic_name}
Context: {topic_desc}
Number of samples: {count}

Strict Output Requirements:
1. Generate realistic, senior-level coding scenarios where the user asks a task requiring tool use.
2. The assistant MUST output a clean, valid tool call using this exact schema:
   <tool_call>
   {{"name": "tool_name", "arguments": {{"key": "value"}}}}
   </tool_call>
   Followed by clear, professional explanation of why this action was taken.
3. Include error handling scenarios (e.g. command failure, non-zero exit code) and how the assistant reacts.
4. Each sample must have high educational and training signal for fine-tuning an agentic coder.
"""
    elif category == "Reasoning":
        prompt = f"""You are a Principal Software Architect and Systems Theorist creating training data for deep technical reasoning.
Target Topic: {topic_name}
Context: {topic_desc}
Number of samples: {count}

Strict Output Requirements:
1. Formulate complex, non-trivial engineering problems requiring multi-step analytical derivation or root-cause analysis (RCA).
2. Assistant response must demonstrate structured step-by-step thinking, mathematical formulation, and deep architectural reasoning.
3. When referencing code or algorithms, provide complete, runnable implementations with zero omissions.
4. Always address trade-offs, edge cases, failure modes, or proofs explicitly.
"""
    else:
        prompt = f"""You are a Staff Software Engineer creating high-density distillation datasets for systems coding.
Target Topic: {topic_name}
Context: {topic_desc}
Number of samples: {count}

Strict Output Requirements:
1. Generate unique, senior-level, highly educational programming tasks and verified complete solutions.
2. Complete runnable code with zero omissions (no placeholders, no '// implement here', no 'TODO').
3. Wrap all runnable code in standard markdown code blocks (e.g. ```cpp ... ``` or ```python ... ```).
4. Always provide explicit Time Complexity and Space Complexity analysis with Big-O notation.
5. Demonstrate explicit error handling, boundary checks, and robust invariant management.
6. All Python snippets must pass ast.parse(). All C/C++ snippets must have balanced brackets and valid syntax.
"""

    batch = await call_gemini_with_backoff(prompt=prompt, model="gemini-3.5-flash-lite")
    
    valid_entries = []
    for item in batch.samples:
        item.system = sanitize_text(item.system)
        item.instruction = sanitize_text(item.instruction)
        item.response = sanitize_text(item.response)
        
        is_valid, passed_notes, failure_reasons = validate_sample(item, trait=trait)
        if is_valid:
            valid_entries.append({
                "messages": [
                    {"role": "system", "content": item.system},
                    {"role": "user", "content": item.instruction},
                    {"role": "assistant", "content": item.response}
                ]
            })
        else:
            print(f"  [Filtered out 1 sample] Reasons: {', '.join(failure_reasons[:2])}", flush=True)
            
    print(f"[{category}] '{topic_name}': {len(valid_entries)}/{len(batch.samples)} verified passed.", flush=True)
    return valid_entries

async def main():
    parser = argparse.ArgumentParser(description="DistillLab Dataset Expander")
    parser.add_argument("--prog-per-topic", type=int, default=25, help="Samples per programming topic (10 topics * 25 = 250)")
    parser.add_argument("--reason-per-topic", type=int, default=25, help="Samples per reasoning topic (5 topics * 25 = 125)")
    parser.add_argument("--tool-per-topic", type=int, default=15, help="Samples per tool use topic (5 topics * 15 = 75)")
    parser.add_argument("--batch-chunk", type=int, default=5, help="API query batch size chunk (default 5)")
    args = parser.parse_args()

    print("======================================================================", flush=True)
    print("  DistillLab -> High-Throughput Multi-Domain Expansion for 'batchcode'", flush=True)
    print(f"  Target Programming: ~{len(PROGRAMMING_TOPICS) * args.prog_per_topic} samples", flush=True)
    print(f"  Target Reasoning:   ~{len(REASONING_TOPICS) * args.reason_per_topic} samples", flush=True)
    print(f"  Target Tool Use:    ~{len(TOOL_USE_TOPICS) * args.tool_per_topic} samples", flush=True)
    print(f"  Total Expansion:    ~{len(PROGRAMMING_TOPICS)*args.prog_per_topic + len(REASONING_TOPICS)*args.reason_per_topic + len(TOOL_USE_TOPICS)*args.tool_per_topic} samples", flush=True)
    print("======================================================================", flush=True)

    total_added = 0

    # 1. Generate Programming & Syntax
    print("\n>>> PHASE 1: Programming & Syntax Datasets", flush=True)
    for name, desc in PROGRAMMING_TOPICS:
        needed = args.prog_per_topic
        retries_left = 3
        while needed > 0 and retries_left > 0:
            chunk = min(needed, args.batch_chunk)
            entries = await distill_topic_batch(name, desc, chunk, "Programming")
            if entries:
                added = commit_to_jsonl(DATASET_LOCAL, entries)
                if os.path.exists(os.path.dirname(DATASET_DOWNLOADS)):
                    commit_to_jsonl(DATASET_DOWNLOADS, entries)
                total_added += added
                needed -= len(entries)
            else:
                retries_left -= 1
                await asyncio.sleep(2)
            await asyncio.sleep(1.2)

    # 2. Generate Reasoning & Problem Decomposition
    print("\n>>> PHASE 2: Technical Reasoning & Architectural Decomposition", flush=True)
    for name, desc in REASONING_TOPICS:
        needed = args.reason_per_topic
        retries_left = 3
        while needed > 0 and retries_left > 0:
            chunk = min(needed, args.batch_chunk)
            entries = await distill_topic_batch(name, desc, chunk, "Reasoning")
            if entries:
                added = commit_to_jsonl(DATASET_LOCAL, entries)
                if os.path.exists(os.path.dirname(DATASET_DOWNLOADS)):
                    commit_to_jsonl(DATASET_DOWNLOADS, entries)
                total_added += added
                needed -= len(entries)
            else:
                retries_left -= 1
                await asyncio.sleep(2)
            await asyncio.sleep(1.2)

    # 3. Generate Tool Use & Agentic Calling
    print("\n>>> PHASE 3: Tool Use & Agentic Function Calling", flush=True)
    for name, desc in TOOL_USE_TOPICS:
        needed = args.tool_per_topic
        retries_left = 3
        while needed > 0 and retries_left > 0:
            chunk = min(needed, args.batch_chunk)
            entries = await distill_topic_batch(name, desc, chunk, "Tool Use")
            if entries:
                added = commit_to_jsonl(DATASET_LOCAL, entries)
                if os.path.exists(os.path.dirname(DATASET_DOWNLOADS)):
                    commit_to_jsonl(DATASET_DOWNLOADS, entries)
                total_added += added
                needed -= len(entries)
            else:
                retries_left -= 1
                await asyncio.sleep(2)
            await asyncio.sleep(1.2)

    print("\n" + "="*70, flush=True)
    print(f"  EXPANSION COMPLETE! Added {total_added} verified, sanitized samples.", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
