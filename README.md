Description

This Python-based system automates book content processing by scraping text and screenshots from a web URL (e.g., https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1) using Playwright. It employs a simulated LLM for content spinning and reviewing, ChromaDB for versioning and semantic search, and a simplified RL-based reward model for content evaluation and retrieval. The system supports human-in-the-loop iterations with user feedback and provides voice output via pyttsx3. Deliverables include the script, screenshots, a demo video, and this public GitHub repository.

Setup





Install Python 3.11 or higher.



Install dependencies:

pip install playwright chromadb sentence-transformers pyttsx3 numpy nltk



Install Playwright browsers:

playwright install



Run the script:

python task.pyNotes





Uses a simulated LLM as no API was provided.



Falls back to ChromaDB’s default embedding if sentence-transformers fails (due to persistent detection issue).



Includes error handling for robust execution.



Screenshots are saved in the screenshots folder.



ChromaDB stores content versions; query with collection.get() or collection.query().

License

This project is for evaluation purposes only, as per the assignment guidelines. The developer retains the license, and Soft-Nerve has no commercial interest.
