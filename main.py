"""FastAPI wrapper for SchemeScout.

Run locally:  uvicorn main:app --reload
Docs:         http://127.0.0.1:8000/docs
"""

import logging
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent import DISCLAIMER, OUTPUT_DIR, run_agent

log = logging.getLogger("schemescout.api")

app = FastAPI(title="SchemeScout", version="1.0.0")


class Query(BaseModel):
    message: str = Field(..., min_length=3, max_length=1000)


@app.get("/")
def health():
    return {"status": "ok", "service": "schemescout"}


@app.post("/chat")
def chat(q: Query):
    try:
        out = run_agent(q.message)
    except Exception:
        log.exception("Agent failed")
        raise HTTPException(status_code=500, detail="Agent failed. Please try again.")

    pdf_path = out.get("pdf_path")
    return {
        "response": out.get("explanation", ""),
        "needs_more_info": out.get("missing", []),
        "profile": out.get("profile"),
        "matched_schemes": [
            {"name": m["name"], "source": m["source"]} for m in out.get("matches", [])
        ],
        "review": out.get("review"),
        "pdf_url": f"/pdf/{Path(pdf_path).stem}" if pdf_path else None,
        "disclaimer": DISCLAIMER,
        "trace": out.get("trace", []),
    }


@app.get("/pdf/{file_id}")
def get_pdf(file_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", file_id):
        raise HTTPException(status_code=400, detail="Invalid file id")
    path = OUTPUT_DIR / f"{file_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path, media_type="application/pdf", filename="scheme_summary.pdf")
