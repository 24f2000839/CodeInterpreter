import os
import sys
import traceback
from io import StringIO
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


# -----------------------------------
# 1. Execute Python Code
# -----------------------------------
def execute_python_code(code: str) -> Dict[str, Any]:
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    stdout_buf = StringIO()
    stderr_buf = StringIO()

    sys.stdout = stdout_buf
    sys.stderr = stderr_buf

    try:
        sandbox_globals = {"__name__": "__main__"}
        exec(code, sandbox_globals)

        out = stdout_buf.getvalue() + stderr_buf.getvalue()
        return {"success": True, "output": out}

    except Exception:
        tb = traceback.format_exc()
        out = stdout_buf.getvalue() + stderr_buf.getvalue() + tb
        return {"success": False, "output": out}

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# -----------------------------------
# 2. Structured Error Analysis
# -----------------------------------
class ErrorAnalysis(BaseModel):
    error_lines: List[int]

def _fallback_extract_lines(tb_text: str) -> List[int]:
    import re

    matches = re.findall(r'File "<string>", line (\d+)', tb_text)

    if not matches:
        return []

    # Return ONLY the last traceback frame (actual error line)
    return [int(matches[-1])]

def analyze_error_with_ai(code: str, tb_text: str) -> List[int]:
    api_key = os.environ.get("GEMINI_API_KEY")

    # Fallback if no API key
    if not api_key:
        return _fallback_extract_lines(tb_text)

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
Analyze this Python code and its traceback.
Return the line number(s) where the error occurred.
Return only JSON: {{ "error_lines": [ ... ] }}

CODE:
{code}

TRACEBACK:
{tb_text}
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "error_lines": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.INTEGER),
                        )
                    },
                    required=["error_lines"],
                ),
            ),
        )

        parsed = ErrorAnalysis.model_validate_json(response.text)

        seen = set()
        uniq = []
        for x in parsed.error_lines:
            if x not in seen:
                seen.add(x)
                uniq.append(x)

        return uniq

    except Exception:
        return _fallback_extract_lines(tb_text)


# -----------------------------------
# 3. Endpoint
# -----------------------------------
@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(req: CodeRequest) -> CodeResponse:
    run = execute_python_code(req.code)

    if run["success"]:
        return CodeResponse(error=[], result=run["output"])

    error_lines = analyze_error_with_ai(req.code, run["output"])
    return CodeResponse(error=error_lines, result=run["output"])
