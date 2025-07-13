from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import asyncio
from openai import AsyncOpenAI

# Initialize AsyncOpenAI client with environment variable
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key"))

app = FastAPI()

class Flashcard(BaseModel):
    question: str
    answer: str

class FlashcardInput(BaseModel):
    text: str

# Chunk long text into smaller ~100-word blocks
def chunk_text(text: str, words_per_chunk: int = 100):
    words = text.split()
    return [' '.join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]

# Estimate importance using keywords
def estimate_importance_score(chunk: str) -> float:
    keywords = ['climate', 'geopolitics', 'conflict', 'development', 'population', 'resources']
    score = sum(1 for word in keywords if word in chunk.lower())
    return min(1.0, 0.3 + 0.1 * score)

# Decide number of flashcards based on importance
def decide_flashcard_count(score: float) -> int:
    if score <= 0.4:
        return 3
    elif score <= 0.6:
        return 4
    elif score <= 0.8:
        return 5
    else:
        return 6

# Async GPT-based flashcard generation
async def gpt_generate_flashcards(chunk: str, n: int) -> List[dict]:
    prompt = f"""
    You are a UPSC flashcard generator.
    Generate {n} flashcards from the passage below:

    \"\"\"{chunk}\"\"\"

    Format each as:
    Q: Question?
    A: Answer.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        output = response.choices[0].message.content
    except Exception as e:
        return [{"question": "Error occurred", "answer": str(e)}]

    # Parse Q&A pairs
    cards = []
    pairs = output.split("Q:")
    for pair in pairs[1:]:
        q, *a_parts = pair.strip().split("A:")
        a = "A:".join(a_parts).strip() if a_parts else ""
        cards.append({"question": q.strip(), "answer": a.strip()})

    return cards

@app.post("/generate-flashcards")
async def generate_flashcards(data: FlashcardInput):
    chunks = chunk_text(data.text, words_per_chunk=100)
    all_flashcards = []

    tasks = []
    for chunk in chunks:
        score = estimate_importance_score(chunk)
        n = decide_flashcard_count(score)
        tasks.append(gpt_generate_flashcards(chunk, n))

    results = await asyncio.gather(*tasks)
    for flashcards in results:
        all_flashcards.extend(flashcards)

    return {"flashcards": all_flashcards}
