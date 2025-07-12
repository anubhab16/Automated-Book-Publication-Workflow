import asyncio
import os
from playwright.async_api import async_playwright
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
import nltk
from nltk.tokenize import sent_tokenize
import pyttsx3
import numpy as np
import uuid

# Debug: Verify sentence-transformers import
try:
    from sentence_transformers import SentenceTransformer
    print("Sentence Transformers imported successfully")
except ImportError as e:
    print(f"Failed to import sentence_transformers: {e}")

# Download NLTK data
nltk.download('punkt')

# Initialize ChromaDB client
client = chromadb.Client()

# Initialize collection with fallback to default embedding if SentenceTransformer fails
try:
    collection = client.create_collection("book_content", embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction())
    print("Using SentenceTransformerEmbeddingFunction for ChromaDB")
except Exception as e:
    print(f"Error with SentenceTransformerEmbeddingFunction: {e}. Falling back to default embedding.")
    collection = client.create_collection("book_content")  # Default embedding function

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Simulated LLM API (no real LLM API needed)
async def llm_api(prompt, role="writer"):
    if role == "writer":
        return f"Spun version of: {prompt[:50]}... (Transformed by AI writer)"
    elif role == "reviewer":
        return f"Reviewed: {prompt[:50]}... Suggestions: Clarify tone, improve flow."
    return prompt

# RL-based reward model (simplified)
def rl_reward(content, metrics={"clarity": 0.7, "coherence": 0.8, "relevance": 0.9}):
    score = sum(metrics.values()) / len(metrics)
    return score

# Scrape content and take screenshot using Playwright
async def scrape_content(url, output_dir="screenshots"):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        content = await page.content()
        screenshot_path = os.path.join(output_dir, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        os.makedirs(output_dir, exist_ok=True)
        await page.screenshot(path=screenshot_path)
        await browser.close()
        return content, screenshot_path

# Store content in ChromaDB with versioning
def store_content(content, chapter_id, version=1):
    doc_id = f"{chapter_id}_v{version}"
    try:
        collection.add(
            documents=[content],
            metadatas=[{"chapter_id": chapter_id, "version": version, "timestamp": str(datetime.now())}],
            ids=[doc_id]
        )
    except Exception as e:
        print(f"Error storing content in ChromaDB: {e}")
    return doc_id

# Semantic search for content retrieval
def search_content(query, n_results=3):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return {"documents": [[]], "metadatas": [[]]}

# RL-based content retrieval
def rl_retrieve_content(query, n_results=3):
    results = search_content(query, n_results)
    best_result = None
    best_score = -1
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        score = rl_reward(doc)
        if score > best_score:
            best_score = score
            best_result = (doc, metadata)
    return best_result

# AI-driven content spinning
async def spin_content(content, chapter_id):
    spun_content = await llm_api(content, role="writer")
    doc_id = store_content(spun_content, chapter_id, version=1)
    return spun_content, doc_id

# AI-driven content review
async def review_content(content):
    review = await llm_api(content, role="reviewer")
    return review

# Human-in-the-loop iteration
def human_review(content, human_input):
    revised_content = f"{content}\nHuman Feedback: {human_input}"
    return revised_content

# Text-to-speech for agentic communication
def speak(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# Main workflow
async def publication_workflow(url, chapter_id, human_inputs=["Looks good, add more detail.", "Finalize content."]):
    # Step 1: Scrape content and take screenshot
    try:
        content, screenshot_path = await scrape_content(url)
        speak(f"Scraped content from {url} and saved screenshot at {screenshot_path}")
        print(f"Scraped content from {url}, screenshot saved at {screenshot_path}")
    except Exception as e:
        print(f"Error scraping content: {e}")
        return

    # Step 2: Store original content in ChromaDB
    doc_id = store_content(content, chapter_id, version=1)
    speak(f"Stored original content with ID {doc_id}")
    print(f"Stored original content with ID {doc_id}")

    # Step 3: AI-driven spinning
    spun_content, spun_doc_id = await spin_content(content, chapter_id)
    speak(f"Generated spun content: {spun_content[:50]}...")
    print(f"Spun content: {spun_content[:50]}... (ID: {spun_doc_id})")

    # Step 4: Human-in-the-loop iterations
    current_content = spun_content
    version = 2
    for human_input in human_inputs:
        # AI review
        review = await review_content(current_content)
        speak(f"AI review: {review[:50]}...")
        print(f"AI review: {review[:50]}...")

        # Human feedback
        current_content = human_review(current_content, human_input)
        doc_id = store_content(current_content, chapter_id, version=version)
        speak(f"Incorporated human feedback, new version: {doc_id}")
        print(f"Incorporated human feedback, new version: {doc_id}")
        version += 1

    # Step 5: RL-based retrieval for final content
    final_content, metadata = rl_retrieve_content(f"chapter_id:{chapter_id}")
    if final_content:
        speak(f"Final content retrieved with score {rl_reward(final_content)}")
        print(f"Final content retrieved: {final_content[:50]}... (Metadata: {metadata})")
    else:
        print("No content retrieved from ChromaDB")

# Run the workflow
if __name__ == "__main__":
    url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
    chapter_id = str(uuid.uuid4())
    asyncio.run(publication_workflow(url, chapter_id))