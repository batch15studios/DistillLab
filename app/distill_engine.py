import ast
import asyncio
import json
import os
import random
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override=True)

class SFTEntry(BaseModel):
    system: str = Field(description="Role, persona guidance, and system context tailored for qwen2.5-coder fine-tuning")
    instruction: str = Field(description="The detailed user coding problem or question")
    response: str = Field(description="Complete verified answer containing runnable code in markdown fences, complexity analysis, and error handling")

class SFTBatch(BaseModel):
    samples: List[SFTEntry]

def estimate_tokens(text: str) -> int:
    """Estimates token length using character count / 3.8 approximation."""
    if not text:
        return 0
    return max(1, round(len(text) / 3.8))

def sanitize_text(text: str) -> str:
    """Strips invalid preprocessor markers caused by markdown headers or LLM formatting anomalies."""
    if not text:
        return text
    # Strip invalid preprocessor markers (#template -> template, #requires -> requires, #concept -> concept)
    cleaned = text.replace("#template", "template").replace("#requires", "requires").replace("#concept", "concept")
    # Also handle regex for any # directly before C++ keywords
    cleaned = re.sub(r'#\s*(template\b)', r'\1', cleaned)
    cleaned = re.sub(r'#\s*(requires\b)', r'\1', cleaned)
    cleaned = re.sub(r'#\s*(concept\b)', r'\1', cleaned)
    return cleaned

def is_cpp_snippet(text: str) -> bool:
    return bool(re.search(r"```(?:c\+\+|cpp|c|h|hpp)\b", text, re.IGNORECASE)) or "#include" in text

def validate_python_code(text: str) -> Tuple[bool, List[str]]:
    """Verifies that all Python snippets parse without syntax errors."""
    pattern = r"```(?:python|py)[^\r\n]*\r?\n([\s\S]*?)```"
    python_blocks = re.findall(pattern, text, re.IGNORECASE)
    
    if python_blocks:
        notes = []
        for idx, block in enumerate(python_blocks, 1):
            try:
                ast.parse(block)
                notes.append(f"Python block #{idx} passed ast.parse()")
            except SyntaxError as e:
                return False, [f"Python block #{idx} SyntaxError at line {e.lineno}: {e.msg}"]
            except Exception as e:
                return False, [f"Python block #{idx} parse error: {str(e)}"]
        return True, notes

    # If no fences, but this is clearly raw Python code (and not C/C++):
    if not is_cpp_snippet(text) and ("def " in text or ("class " in text and ":" in text and "{" not in text)):
        try:
            ast.parse(text)
            return True, ["Raw Python code verified via ast.parse()"]
        except SyntaxError as e:
            return False, [f"Python SyntaxError at line {e.lineno}: {e.msg}"]

    return True, ["No Python code blocks detected"]

def validate_cpp_code(text: str) -> Tuple[bool, List[str]]:
    """Checks basic bracket/brace and structural integrity for C/C++ snippets."""
    pattern = r"```(?:c\+\+|cpp|c|h|hpp)[^\r\n]*\r?\n([\s\S]*?)```"
    cpp_blocks = re.findall(pattern, text, re.IGNORECASE)
    
    if not cpp_blocks and is_cpp_snippet(text):
        # Extract code portion before markdown explanation headers
        code_part = re.split(r"\n(?=#{1,4}\s|\*\*Complexity|\*\*Time|###|Complexity:)", text)[0]
        cpp_blocks = [code_part]

    if not cpp_blocks:
        return True, ["No C/C++ code blocks detected"]

    notes = []
    for idx, code in enumerate(cpp_blocks, 1):
        cleaned = []
        i = 0
        n = len(code)
        in_single = False
        in_multi = False
        in_str = False
        in_chr = False
        escape = False

        while i < n:
            c = code[i]
            nxt = code[i+1] if i + 1 < n else ''

            if in_single:
                if c == '\n':
                    in_single = False
                    cleaned.append(c)
                i += 1
                continue
            if in_multi:
                if c == '*' and nxt == '/':
                    in_multi = False
                    i += 2
                    continue
                i += 1
                continue
            if in_str:
                if escape:
                    escape = False
                elif c == '\\':
                    escape = True
                elif c == '"':
                    in_str = False
                i += 1
                continue
            if in_chr:
                if escape:
                    escape = False
                elif c == '\\':
                    escape = True
                elif c == "'":
                    in_chr = False
                i += 1
                continue

            if c == '/' and nxt == '/':
                in_single = True
                i += 2
                continue
            elif c == '/' and nxt == '*':
                in_multi = True
                i += 2
                continue
            elif c == '"':
                in_str = True
                i += 1
                continue
            elif c == "'":
                in_chr = True
                i += 1
                continue

            cleaned.append(c)
            i += 1

        clean_text = "".join(cleaned)
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        for ch in clean_text:
            if ch in '({[':
                stack.append(ch)
            elif ch in ')}]':
                if not stack or stack[-1] != pairs[ch]:
                    return False, [f"C/C++ block #{idx} has mismatched bracket '{ch}'"]
                stack.pop()

        if stack:
            return False, [f"C/C++ block #{idx} has unclosed bracket '{stack[-1]}'"]

        if len(code.strip()) > 40 and ';' not in clean_text:
            return False, [f"C/C++ block #{idx} appears to be missing statements or semicolons"]

        notes.append(f"C/C++ block #{idx} verified: balanced brackets & valid structure")
    return True, notes

def validate_complexity(text: str) -> Tuple[bool, str]:
    """Checks for presence of explicit time and space computational complexity analysis."""
    lower = text.lower()
    has_time = bool(re.search(r"\b(time\s+complexity|temporal|runtime\s+complexity|time:\s*o\(|run-time)\b", lower))
    has_space = bool(re.search(r"\b(space\s+complexity|spatial|memory\s+complexity|auxiliary\s+space|space:\s*o\()\b", lower))
    has_big_o = bool(re.search(r"\b[oO]\([n1kmlog\s\*\+\^\d\w/]+\)", text))

    if (has_time and has_space) or (has_big_o and ("complexity" in lower or "space" in lower or "time" in lower)):
        return True, "Complexity analysis verified (Time & Space)"
    return False, "Missing explicit time and space computational complexity analysis"

def validate_error_handling(text: str) -> Tuple[bool, str]:
    """Verifies that the response incorporates proper error handling or boundary validation."""
    patterns = [
        r"\btry\b[\s\S]+?\b(except|catch)\b",
        r"\braise\s+[A-Z]\w*",
        r"\bthrow\s+[^;]+",
        r"\bassert\b",
        r"\b(nullptr|NULL)\b",
        r"\berrno\b",
        r"\bperror\b",
        r"\bif\s*\([^)]*(==\s*null|< 0|!= 0|!ptr|!fd|is None|not\s|is\s+none|nullptr)",
        r"\b(std::optional|std::expected|Result<|Option<)\b",
        r"\b(ValueError|TypeError|RuntimeError|IndexError|KeyError|std::exception|std::runtime_error|std::invalid_argument|std::out_of_range)\b",
        r"\b(exit\(1\)|return\s+(-1|false|NULL|nullptr|nullptr_t))\b",
        r"error\s+handling|boundary\s+check|validation\s+check|guard\s+clause|error\s+condition",
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, "Error handling and validation verified"
    return False, "Missing proper error handling, exception guards, or boundary validation"

def validate_sample(item: SFTEntry) -> Tuple[bool, List[str], List[str]]:
    """Runs the complete validation pipeline on a single distilled sample."""
    passed_notes = []
    failure_reasons = []

    # 1. Python AST parse check
    py_ok, py_notes = validate_python_code(item.response)
    if py_ok:
        passed_notes.extend(py_notes)
    else:
        failure_reasons.extend(py_notes)

    # 2. C/C++ bracket & syntax integrity
    cpp_ok, cpp_notes = validate_cpp_code(item.response)
    if cpp_ok:
        passed_notes.extend(cpp_notes)
    else:
        failure_reasons.extend(cpp_notes)

    # 3. Complexity analysis check
    comp_ok, comp_note = validate_complexity(item.response)
    if comp_ok:
        passed_notes.append(comp_note)
    else:
        failure_reasons.append(comp_note)

    # 4. Error handling check
    err_ok, err_note = validate_error_handling(item.response)
    if err_ok:
        passed_notes.append(err_note)
    else:
        failure_reasons.append(err_note)

    # 5. Non-omission check
    if re.search(r"(//|#)\s*(implement here|todo|fill in|omitted|left as an exercise)", item.response, re.IGNORECASE):
        failure_reasons.append("Detected omission placeholder ('// implement here' / 'TODO')")

    is_valid = len(failure_reasons) == 0
    return is_valid, passed_notes, failure_reasons

def get_genai_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key and key.strip():
        return genai.Client(api_key=key.strip())
    return genai.Client()

async def call_gemini_with_backoff(
    prompt: str,
    model: str = "gemini-3.5-flash-lite",
    api_key: Optional[str] = None,
    max_retries: int = 4,
    base_delay: float = 1.5,
) -> SFTBatch:
    """Calls Gemini API with exponential backoff on transient/rate-limit errors."""
    client = get_genai_client(api_key)
    last_error = None

    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SFTBatch,
                    temperature=0.7,
                )
            )
            data = json.loads(response.text)
            return SFTBatch(**data)
        except Exception as e:
            last_error = e
            err_msg = str(e).lower()
            
            # If 2.5 deprecated, fall back to current 3.5/3.8 model
            if "not available to new users" in err_msg or "not found" in err_msg:
                try:
                    fallback_model = "gemini-3.5-flash-lite"
                    response = await client.aio.models.generate_content(
                        model=fallback_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=SFTBatch,
                            temperature=0.7,
                        )
                    )
                    data = json.loads(response.text)
                    return SFTBatch(**data)
                except Exception as inner_e:
                    last_error = inner_e

            is_transient = any(term in err_msg for term in ["429", "resource_exhausted", "503", "unavailable", "500", "timeout", "deadline"])
            if attempt < max_retries - 1 and is_transient:
                sleep_time = (base_delay ** (attempt + 1)) + random.uniform(0.1, 0.5)
                await asyncio.sleep(sleep_time)
                continue
            break

    raise RuntimeError(f"Gemini API request failed: {last_error}")

async def run_distillation(
    topic: str,
    count: int = 5,
    custom_prompt: Optional[str] = None,
    model: str = "gemini-3.5-flash-lite",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs distillation, applies the live validation pipeline, and formats to ShareGPT/OpenAI messages schema."""
    target_instructions = """Strict Output Requirements for Fine-Tuning qwen2.5-coder:1.5b:
1. Generate unique, senior-level, highly educational programming tasks and verified complete solutions.
2. Complete runnable code with zero omissions (no placeholders, no '// implement here', no 'TODO').
3. Wrap all runnable code in standard markdown code blocks (e.g. ```cpp ... ``` or ```python ... ```).
4. Always provide explicit Time Complexity and Space Complexity analysis with Big-O notation.
5. Demonstrate explicit error handling, boundary checks, and robust invariant management.
6. All Python snippets must pass ast.parse(). All C/C++ snippets must have balanced brackets and valid syntax.
"""

    if custom_prompt and custom_prompt.strip():
        prompt = f"""You are a Staff Software Engineer creating a high-density distillation dataset for fine-tuning qwen2.5-coder:1.5b.
Custom focus: {custom_prompt.strip()}
Primary Topic domain: {topic}
Number of samples to generate: {count}

{target_instructions}
"""
    else:
        prompt = f"""You are a Staff Software Engineer creating a high-density distillation dataset for fine-tuning qwen2.5-coder:1.5b.
Target Topic: {topic}
Number of samples to generate: {count}

{target_instructions}
"""

    batch = await call_gemini_with_backoff(prompt=prompt, model=model, api_key=api_key)

    valid_samples = []
    rejected_samples = []

    for item in batch.samples:
        # Pre-sanitize text to eliminate markdown preprocessor anomalies
        item.system = sanitize_text(item.system)
        item.instruction = sanitize_text(item.instruction)
        item.response = sanitize_text(item.response)

        is_valid, passed_notes, failure_reasons = validate_sample(item)
        full_text = f"{item.system} {item.instruction} {item.response}"
        tokens = estimate_tokens(full_text)
        sample_id = str(uuid.uuid4())

        sample_record = {
            "id": sample_id,
            "topic": topic,
            "messages": [
                {"role": "system", "content": item.system},
                {"role": "user", "content": item.instruction},
                {"role": "assistant", "content": item.response}
            ],
            "estimated_tokens": tokens,
            "is_overflow": tokens > 4096,
            "is_valid": is_valid,
            "validation_notes": passed_notes,
            "failure_reasons": failure_reasons
        }

        if is_valid:
            valid_samples.append(sample_record)
        else:
            rejected_samples.append(sample_record)

    return {
        "topic": topic,
        "model": model,
        "total_requested": count,
        "total_generated": len(batch.samples),
        "valid_count": len(valid_samples),
        "rejected_count": len(rejected_samples),
        "valid_samples": valid_samples,
        "rejected_samples": rejected_samples
    }

def commit_to_jsonl(filepath: str, entries: List[Dict[str, Any]]) -> int:
    """Appends valid entries to the local .jsonl file in ShareGPT/OpenAI messages format with sanitization."""
    if not entries:
        return 0
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    count = 0
    with open(filepath, "a", encoding="utf-8") as f:
        for entry in entries:
            raw_msgs = entry.get("messages", [])
            sanitized_msgs = []
            for m in raw_msgs:
                sanitized_msgs.append({
                    "role": m.get("role", ""),
                    "content": sanitize_text(m.get("content", ""))
                })
            formatted = {
                "messages": sanitized_msgs
            }
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            count += 1
    return count
