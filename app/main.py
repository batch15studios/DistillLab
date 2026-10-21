import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.distill_engine import (
    run_distillation,
    commit_to_jsonl,
    estimate_tokens,
    sanitize_text,
)

load_dotenv()

app = FastAPI(title="DistillLab", description="Dataset Distillation Dashboard")

DATASET_PATH = "data/cs_programming_dataset.jsonl"
TEMPLATE_PATH = Path("app/templates/index.html")

TOPIC_PRESETS = [
    # Coding Presets
    {
        "id": "systems_cpp",
        "trait": "coding",
        "name": "Systems Programming (C/C++)",
        "topic": "Systems Programming in C/C++ (Memory management, pointers, bitwise manipulation, RAII)",
        "prompt_hint": "Focus on manual memory management, pointer arithmetic, custom allocators, cache-friendly data layouts, and RAII wrappers."
    },
    {
        "id": "concurrency",
        "trait": "coding",
        "name": "Concurrency & Multi-Threading",
        "topic": "Concurrency & Multi-Threading (Lock-free queues, atomic operations, mutexes, memory barriers)",
        "prompt_hint": "Focus on lock-free data structures, atomic memory order (acquire/release/relaxed), condition variables, and deadlock-free designs."
    },
    {
        "id": "graph_theory",
        "trait": "coding",
        "name": "Algorithmic Graph Theory",
        "topic": "Algorithmic Graph Theory (Tarjan's SCC, Dinic's Max Flow, Shortest Paths, Topological Sort)",
        "prompt_hint": "Focus on production-grade graph algorithms, adjacency lists, cycle detection, strongly connected components, and min-cut/max-flow."
    },
    {
        "id": "os_internals",
        "trait": "coding",
        "name": "OS Internals & Kernels",
        "topic": "OS Internals & Kernel Subsystems (Virtual memory paging, process scheduling, IPC, system calls)",
        "prompt_hint": "Focus on page table traversal, epoll/kqueue event loops, fork/exec, shared memory IPC, and signal handling in POSIX systems."
    },
    {
        "id": "db_engines",
        "trait": "coding",
        "name": "Database Storage Engines",
        "topic": "Database Storage Engines (B+ Trees, LSM-Trees, Write-Ahead Logging (WAL), Buffer Pools)",
        "prompt_hint": "Focus on B+ tree node splitting/merging, SSTable compaction, append-only commit logs, and LRU/Clock buffer pool management."
    },
    # Reasoning Presets
    {
        "id": "concurrency_rca",
        "trait": "reasoning",
        "name": "Concurrency Bug & Race RCA",
        "topic": "Concurrency Bug Analysis & Race Condition RCA (Deadlocks, ABA problems, lock inversion, memory visibility)",
        "prompt_hint": "Provide deep root-cause analysis, step-by-step trace of concurrent thread schedules, invariant proofs, and remediation."
    },
    {
        "id": "cache_coherence",
        "trait": "reasoning",
        "name": "Cache Coherence & Architecture",
        "topic": "Hardware Architecture & Cache Coherence (L1/L2/L3 cache lines, false sharing, MESI protocol, memory bus contention)",
        "prompt_hint": "Analyze cache-line bouncing, memory layout alignment tradeoffs, and performance degradation causes."
    },
    {
        "id": "distributed_consensus",
        "trait": "reasoning",
        "name": "Distributed Consensus Proofs",
        "topic": "Distributed Consensus & State Machine Replication (Raft election safety, log matching invariants, split-brain proofs)",
        "prompt_hint": "Focus on rigorous proof of quorum intersections, election timeout dynamics, and partition recovery."
    },
    # Tool-Use Presets
    {
        "id": "terminal_agent",
        "trait": "tool_use",
        "name": "Terminal & Shell Tool Execution",
        "topic": "Agentic Terminal Execution (Running commands, parsing grep/find output, error diagnosis, verifying exit codes)",
        "prompt_hint": "Include structured <tool_call>{\"name\": \"run_command\", \"arguments\": {\"command\": \"...\"}}</tool_call> and result handling."
    },
    {
        "id": "code_search_agent",
        "trait": "tool_use",
        "name": "Codebase Navigation & AST Search",
        "topic": "Code Search & Multi-File Navigation (Finding symbol definitions, grepping references, reading file slices)",
        "prompt_hint": "Demonstrate selecting appropriate search tools, reasoning about match sets, and locating target implementations."
    },
    {
        "id": "git_workflow_agent",
        "trait": "tool_use",
        "name": "Git & Diff Patching Agent",
        "topic": "Git Operations & Conflict Resolution (Inspecting git diff, branch management, staging atomic changes)",
        "prompt_hint": "Demonstrate tool calls for checking git status, inspecting diff chunks, and applying surgical patches."
    }
]

class GenerateRequest(BaseModel):
    topic: str
    count: int = Field(default=5, ge=1, le=50)
    custom_prompt: Optional[str] = None
    trait: str = Field(default="coding", description="Trait: coding, reasoning, tool_use, general")
    model: str = "gemini-3.5-flash-lite"
    auto_commit: bool = False
    api_key: Optional[str] = None

class CommitRequest(BaseModel):
    samples: List[Dict[str, Any]]

class KeyConfigRequest(BaseModel):
    api_key: str
    save_to_env: bool = True

@app.get("/api/presets")
def get_presets():
    return TOPIC_PRESETS

@app.get("/api/config/key-status")
def key_status():
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return {
        "is_configured": bool(env_key and env_key.strip()),
        "source": "environment" if env_key else "missing"
    }

@app.post("/api/config/key")
def configure_key(req: KeyConfigRequest):
    cleaned = req.api_key.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="API Key cannot be empty.")
    os.environ["GEMINI_API_KEY"] = cleaned
    if req.save_to_env:
        env_path = ".env"
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = [line for line in f if not line.startswith("GEMINI_API_KEY=")]
        lines.append(f"GEMINI_API_KEY={cleaned}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return {"status": "success", "message": "API Key saved successfully."}

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    try:
        result = await run_distillation(
            topic=req.topic,
            count=req.count,
            custom_prompt=req.custom_prompt,
            trait=req.trait,
            model=req.model,
            api_key=req.api_key
        )
        committed = 0
        if req.auto_commit and result["valid_samples"]:
            committed = commit_to_jsonl(DATASET_PATH, result["valid_samples"])

        return {
            "status": "success",
            "topic": result["topic"],
            "model": result["model"],
            "total_requested": result["total_requested"],
            "total_generated": result["total_generated"],
            "valid_count": result["valid_count"],
            "rejected_count": result["rejected_count"],
            "valid_samples": result["valid_samples"],
            "rejected_samples": result["rejected_samples"],
            "committed": committed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/samples")
def list_samples(offset: int = 0, limit: int = 50, search: Optional[str] = None):
    if not os.path.exists(DATASET_PATH):
        return {"total": 0, "samples": []}

    entries = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                full_text = " ".join([m.get("content", "") for m in data.get("messages", [])])
                tokens = estimate_tokens(full_text)
                
                if search and search.strip():
                    q = search.strip().lower()
                    if q not in full_text.lower():
                        continue

                entries.append({
                    "index": idx,
                    "messages": data.get("messages", []),
                    "estimated_tokens": tokens
                })
            except Exception:
                continue

    total = len(entries)
    paginated = entries[offset: offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "samples": paginated
    }

@app.post("/api/samples/commit")
def commit_samples(req: CommitRequest):
    added = commit_to_jsonl(DATASET_PATH, req.samples)
    stats_data = get_stats()
    return {
        "status": "success",
        "committed": added,
        "total_samples": stats_data["total_samples"],
        "file_size_kb": stats_data["file_size_kb"]
    }

@app.delete("/api/samples/{index}")
def delete_sample(index: int):
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=404, detail="Dataset file not found.")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if index < 0 or index >= len(lines):
        raise HTTPException(status_code=400, detail=f"Invalid index {index}. Total lines: {len(lines)}")

    lines.pop(index)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return {"status": "success", "remaining": len(lines)}

@app.post("/api/samples/clear")
def clear_samples():
    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
    return {"status": "success", "message": "Dataset cleared."}

@app.post("/api/dataset/sanitize")
def sanitize_dataset_endpoint():
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    occ_template = content.count("#template")
    occ_requires = content.count("#requires")
    occ_concept = content.count("#concept")
    total_found = occ_template + occ_requires + occ_concept

    cleaned = sanitize_text(content)

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        f.write(cleaned)

    stats = get_stats()
    return {
        "status": "success",
        "fixed_occurrences": total_found,
        "details": {
            "#template": occ_template,
            "#requires": occ_requires,
            "#concept": occ_concept
        },
        "stats": stats
    }

@app.get("/api/export")
def export_dataset():
    if not os.path.exists(DATASET_PATH) or os.path.getsize(DATASET_PATH) == 0:
        raise HTTPException(status_code=404, detail="Dataset is empty or does not exist yet.")
    return FileResponse(
        DATASET_PATH,
        media_type="application/x-jsonlines",
        filename="cs_programming_dataset.jsonl"
    )

@app.get("/api/stats")
def get_stats():
    if not os.path.exists(DATASET_PATH):
        return {
            "total_samples": 0,
            "file_size_kb": 0.0,
            "total_tokens": 0,
            "avg_tokens": 0,
            "filepath": DATASET_PATH
        }

    total_tokens = 0
    count = 0
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                full_text = " ".join([m.get("content", "") for m in data.get("messages", [])])
                total_tokens += estimate_tokens(full_text)
                count += 1
            except Exception:
                continue

    file_size_kb = round(os.path.getsize(DATASET_PATH) / 1024, 2)
    avg_tokens = round(total_tokens / count) if count > 0 else 0

    return {
        "total_samples": count,
        "file_size_kb": file_size_kb,
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "filepath": DATASET_PATH
    }

@app.get("/", response_class=HTMLResponse)
def index():
    if not TEMPLATE_PATH.exists():
        return HTMLResponse("<h1>DistillLab loading...</h1>", status_code=200)
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

