from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Knowledge base ───────────────────
knowledge_base = [
    # Python / Software Development
    "FastAPI allows developers to build REST APIs using Python.",
    "Pydantic validates incoming API request data.",
    "SQLAlchemy connects Python applications to relational databases.",
    "Docker containers help deploy Python applications.",
    "Git tracks changes in software development projects.",

    # Cooking
    "Pasta should be cooked in boiling salted water.",
    "Baking bread requires flour, yeast, water, and time.",
    "Grilling vegetables enhances their natural flavor.",
    "A chef uses a sharp knife to prepare ingredients.",
    "Recipes provide instructions for preparing meals.",

    # Space
    "Mars is the fourth planet from the Sun.",
    "Astronauts travel into space aboard spacecraft.",
    "NASA launches rockets to explore the solar system.",
    "The Moon orbits Earth approximately every 27 days.",
    "Telescopes allow astronomers to observe distant galaxies.",

    # Music
    "Guitars produce sound when their strings vibrate.",
    "Pianos use keys to produce musical notes.",
    "Musicians practice scales to improve their technique.",
    "A symphony orchestra contains strings, brass, and percussion.",
    "Music producers use software to record and edit audio."
]

# ── Test queries ───────────────

test_queries = [
    "How can I deploy my Python application?",
    "What ingredients do I need to make dinner?",
    "How do astronauts explore other planets?",
    "How can I learn to play a musical instrument?",
    "How do I use software to create something?"
]

# ── Thresholds ───────────────────
THRESHOLDS = [0.3, 0.5, 0.7]

# ── Model & embeddings ─────────────
print("Loading model and embedding knowledge base...")

model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(knowledge_base)

# ── Run experiment ────────────
for query in test_queries:
    print(f'\nQuery: "{query}"')

    query_embedding = model.encode([query])
    scores = cosine_similarity(
        query_embedding, doc_embeddings
    )[0]
    for threshold in THRESHOLDS:
        passing = [
            (score, sentence)
            for score, sentence in zip(scores, knowledge_base)
            if score >= threshold
        ]
        passing.sort(reverse=True)

        count = len(passing)
        word = "result" if count == 1 else "results"
        print(f"\n  Threshold {threshold}: {count} {word}")

        for score, sentence in passing:
            print(f"    [{score:.2f}] {sentence}")

        if not passing:
            print("    No results passed this threshold.")

#Identify potentially missed relevant results
        excluded = [
            (score, sentence)
            for score, sentence in zip(scores, knowledge_base)
            if score < threshold
        ]

        excluded.sort(reverse=True)

        if excluded:
            score, sentence = excluded[0]
            print(
                f"    Highest excluded: [{score:.4f}] {sentence}"
            )
            print("      Review whether this result is relevant.")

