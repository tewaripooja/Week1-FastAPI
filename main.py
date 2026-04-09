import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from ddgs import DDGS
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Week 1 FastAPI Research Service")

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    max_results: int = Field(5, ge=1, le=10)


class ResearchItem(BaseModel):
    title: str
    url: str
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    results: List[ResearchItem]
    source_note: Optional[str] = None


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=50000)
    max_bullets: int = Field(5, ge=3, le=12)


class SummarizeResponse(BaseModel):
    summary: str


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=20000)


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: int = Field(..., ge=0, le=100)
    reasoning: str


def _format_bullets(text: str, max_bullets: int) -> str:
    normalized = text.replace("\r\n", "\n").strip()
    parts = re.split(r"\n+|(?:^|\s)[\-\u2022]\s+|\s\d+[.)]\s+", normalized)
    bullets = [p.strip(" -\u2022\t") for p in parts if p.strip(" -\u2022\t")]
    if not bullets:
        return normalized
    return "\n".join([f"- {item}" for item in bullets[:max_bullets]])


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        raw_results = list(DDGS().text(request.question, max_results=request.max_results))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc

    items = [
        ResearchItem(
            title=item.get("title", "").strip() or "Untitled",
            url=item.get("href", "").strip(),
            snippet=item.get("body", "").strip(),
        )
        for item in raw_results
    ]

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is empty in .env")

    sources = "\n".join([f"- {r.title}\n  {r.url}\n  {r.snippet}" for r in items]) or "No sources."
    prompt = (
        f"Question: {request.question}\n\n"
        f"Sources:\n{sources}\n\n"
        "Give a concise answer in bullet points. Use only the sources."
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model="gpt-5-nano", input=prompt)
        answer = (response.output_text or "").strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI failed: {exc}") from exc

    source_note = "No sources found." if not items else None
    return AskResponse(
        question=request.question,
        answer=answer,
        results=items,
        source_note=source_note,
    )


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is empty in .env")

    prompt = (
        "Summarize the following text into clear bullet points.\n"
        f"Use at most {request.max_bullets} bullets and keep each bullet short.\n"
        "Each bullet must be on a separate new line.\n\n"
        f"Text:\n{request.text}"
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model="gpt-5-nano", input=prompt)
        summary = (response.output_text or "").strip()
        if not summary:
            raise HTTPException(status_code=502, detail="OpenAI returned an empty summary")
        summary = _format_bullets(summary, request.max_bullets)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI failed: {exc}") from exc

    return SummarizeResponse(summary=summary)


@app.post("/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest) -> SentimentResponse:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is empty in .env")

    prompt = (
        "Analyze sentiment for the text below.\n"
        "Return strict JSON only with keys: sentiment, confidence, reasoning.\n"
        "sentiment must be one of: positive, negative, neutral, mixed.\n"
        "confidence must be an integer from 0 to 100.\n"
        "reasoning must be a short one-sentence explanation.\n\n"
        f"Text:\n{request.text}"
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model="gpt-5-nano", input=prompt)
        raw_output = (response.output_text or "").strip()
        if not raw_output:
            raise HTTPException(status_code=502, detail="OpenAI returned an empty sentiment result")
        parsed = SentimentResponse.model_validate_json(raw_output)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Sentiment analysis failed: {exc}") from exc

    if parsed.sentiment not in {"positive", "negative", "neutral", "mixed"}:
        raise HTTPException(status_code=502, detail="Invalid sentiment label returned by model")

    return parsed
