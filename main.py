import os
import cognee
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

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