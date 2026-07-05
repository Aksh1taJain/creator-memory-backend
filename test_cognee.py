import asyncio
import os

import cognee
from dotenv import load_dotenv

load_dotenv()


async def main():

    await cognee.serve(
        url=os.getenv("COGNEE_SERVICE_URL"),
        api_key=os.getenv("COGNEE_API_KEY")
    )

    print("Connected to Cognee Cloud!")

    await cognee.remember(
        "The creator prefers fast-paced edits and uploads videos at 7 PM."
    )

    print("Memory stored!")

    results = await cognee.recall(
        "When does the creator upload videos?"
    )

    print("\nRecall Results:\n")

    for r in results:
        print(r)

    await cognee.disconnect()


asyncio.run(main())