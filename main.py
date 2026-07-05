import os
import cognee
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

app = FastAPI(title="CreatorMemory Backend")


class RememberRequest(BaseModel):
    userId: str = "demo-user"
    memory: str


class RecallRequest(BaseModel):
    userId: str = "demo-user"
    query: str = "creator preferences and past video performance"


@app.on_event("startup")
async def startup():
    await cognee.serve(
        url=os.getenv("COGNEE_SERVICE_URL"),
        api_key=os.getenv("COGNEE_API_KEY")
    )
    print("Connected to Cognee Cloud")


@app.get("/")
async def root():
    return {"status": "CreatorMemory backend running"}


@app.post("/remember")
async def remember(req: RememberRequest):
    text = f"Creator {req.userId}: {req.memory}"

    await cognee.remember(text)

    return {
        "success": True,
        "message": "Memory saved in Cognee."
    }


@app.post("/recall")
async def recall(req: RecallRequest):
    results = await cognee.recall(req.query)

    memory_text = "\n".join(
        [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in results]
    )

    return {
        "success": True,
        "memory": memory_text
    }

client = OpenAI(
    base_url=os.getenv("LLM_ENDPOINT"),
    api_key=os.getenv("LLM_API_KEY")
)


class OptimizeVideoRequest(BaseModel):
    userId: str = "demo-user"
    topic: str
    transcript: str = ""
    instructions: str = ""


@app.post("/optimize-video")
async def optimize_video(req: OptimizeVideoRequest):
    memory_results = await cognee.recall(
        f"Creator {req.userId} preferences, audience, editing style, upload time, past performance"
    )

    memory_text = "\n".join(
        [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in memory_results]
    )

    prompt = f"""
You are CreatorMemory AI, a YouTube video optimization agent.

Use the creator memory to personalize the recommendations.

CREATOR MEMORY:
{memory_text}

VIDEO TOPIC:
{req.topic}

TRANSCRIPT / SUMMARY:
{req.transcript}

USER INSTRUCTIONS:
{req.instructions}

Generate a polished YouTube optimization plan with:
1. Video vibe analysis
2. Strong opening hook
3. Editing suggestions
4. 5 title options
5. Description
6. Hashtags
7. Thumbnail concept
8. Royalty-free music style
9. Best upload time
10. What should be remembered for future videos

Do not guarantee virality. Be practical and creator-friendly.
"""

    completion = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        messages=[
            {"role": "system", "content": "You are a helpful AI creator assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1200
    )

    answer = completion.choices[0].message.content

    await cognee.remember(
        f"""Creator {req.userId} created a video about: {req.topic}. Instructions/preferences given: {req.instructions}. Remember useful long-term creator preferences from this interaction."""
    )

    return {
        "success": True,
        "memory_used": memory_text,
        "optimization": answer
    }