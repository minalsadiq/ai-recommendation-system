
#   DecodeLabs Industrial Training Kit | Batch 2026
#   Name  : Minal Sadiq
#   PROJECT 3: Tech Stack Recommender
#   Concepts Used: TF-IDF + Cosine Similarity
# ============================================================ #

import math
import csv
import os

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD THE DATASET (raw_skills.csv)
# Each row = one job role + its required skills
# ─────────────────────────────────────────────────────────────

def load_dataset(filepath):
    """
    Reads the CSV file and returns a list of job roles.
    Each item looks like:
      { "role": "Data Scientist", "skills": ["python", "sql", ...] }
    """
    dataset = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            skills_list = row['skills'].strip().lower().split()
            dataset.append({
                "role": row['role'],
                "skills": skills_list
            })
    return dataset


# ─────────────────────────────────────────────────────────────
# STEP 2: BUILD VOCABULARY
# Collect all unique skills across all job roles
# Example: ["python", "sql", "docker", "aws", ...]
# ─────────────────────────────────────────────────────────────

def build_vocabulary(dataset):
    """
    Returns a sorted list of ALL unique skills found in the dataset.
    This becomes our 'shared vocabulary space'.
    """
    vocab = set()
    for item in dataset:
        for skill in item["skills"]:
            vocab.add(skill)
    return sorted(list(vocab))


# ─────────────────────────────────────────────────────────────
# STEP 3: TF-IDF WEIGHTING
# TF  = How often a skill appears in this role / total skills in role
# IDF = log( total roles / roles that have this skill )
# TF-IDF = TF × IDF   (rare+specific skills get higher score)
# ─────────────────────────────────────────────────────────────

def compute_tf(skills_list):
    """
    Term Frequency: count each skill, divide by total skills.
    Example: ["python", "sql", "python"] → {"python": 0.67, "sql": 0.33}
    """
    tf = {}
    total = len(skills_list)
    for skill in skills_list:
        tf[skill] = tf.get(skill, 0) + 1
    for skill in tf:
        tf[skill] = tf[skill] / total
    return tf


def compute_idf(dataset, vocabulary):
    """
    Inverse Document Frequency: punishes skills that appear in EVERY role.
    Common skills like 'git' or 'python' get lower weight.
    Rare skills like 'flutter' or 'ethical_hacking' get higher weight.
    """
    total_docs = len(dataset)
    idf = {}
    for skill in vocabulary:
        # Count how many roles contain this skill
        docs_with_skill = sum(1 for item in dataset if skill in item["skills"])
        # Add 1 to avoid division by zero
        idf[skill] = math.log(total_docs / (1 + docs_with_skill))
    return idf


def compute_tfidf_vector(skills_list, vocabulary, idf):
    """
    Converts a list of skills into a TF-IDF number vector.
    Example: ["python", "sql"] → [0.12, 0, 0.08, 0, 0.31, ...]
             (one number per vocabulary word)
    """
    tf = compute_tf(skills_list)
    vector = []
    for skill in vocabulary:
        tf_val = tf.get(skill, 0)
        idf_val = idf.get(skill, 0)
        vector.append(tf_val * idf_val)
    return vector


# ─────────────────────────────────────────────────────────────
# STEP 4: COSINE SIMILARITY
# Measures the ANGLE between two vectors (0 = no match, 1 = perfect)
# Formula: cos(θ) = (A · B) / (||A|| × ||B||)
# ─────────────────────────────────────────────────────────────

def cosine_similarity(vec_a, vec_b):
    """
    Compares two vectors and returns a score between 0 and 1.
    Score closer to 1.0 = very similar (good match!)
    Score closer to 0.0 = very different (bad match)
    """
    # Dot product: multiply matching positions and add them up
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # Magnitudes (lengths) of each vector
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    # Avoid division by zero (happens when user has no matching skills)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ─────────────────────────────────────────────────────────────
# STEP 5: THE FULL RECOMMENDATION PIPELINE
# Ingestion → Scoring → Sorting → Filtering (Top-N)
# ─────────────────────────────────────────────────────────────

def recommend(user_skills, dataset, vocabulary, idf, top_n=3):
    """
    Main pipeline:
    1. Convert user skills to a TF-IDF vector
    2. Score every job role against the user vector
    3. Sort by score (highest first)
    4. Return only the Top-N results
    """

    # --- Pipeline Step 1: INGESTION ---
    # Clean user input (lowercase, replace spaces with underscore)
    cleaned_skills = [s.strip().lower().replace(" ", "_") for s in user_skills]

    # Build user's TF-IDF vector
    user_vector = compute_tfidf_vector(cleaned_skills, vocabulary, idf)

    # --- Pipeline Step 2: SCORING ---
    scores = []
    for item in dataset:
        role_vector = compute_tfidf_vector(item["skills"], vocabulary, idf)
        score = cosine_similarity(user_vector, role_vector)
        scores.append({"role": item["role"], "score": score, "skills": item["skills"]})

    # --- Pipeline Step 3: SORTING ---
    scores.sort(key=lambda x: x["score"], reverse=True)

    # --- Pipeline Step 4: FILTERING (Top-N) ---
    return scores[:top_n]


# ─────────────────────────────────────────────────────────────
# STEP 6: USER INTERFACE Pretty Input/Output
# ─────────────────────────────────────────────────────────────

def display_banner():
    print("\n" + "=" * 55)
    print("   TECH STACK RECOMMENDER  |  DecodeLabs 2026")
    print("=" * 55)
    print("   Built with: TF-IDF + Cosine Similarity")
    print("=" * 55 + "\n")


def display_results(recommendations, user_skills):
    print("\n" + "─" * 55)
    print(f"    Your Skills : {', '.join(user_skills)}")
    print("─" * 55)
    print("    TOP RECOMMENDED CAREER PATHS:\n")

    medals = ["🥇", "🥈", "🥉"]
    for i, rec in enumerate(recommendations):
        score_percent = round(rec["score"] * 100, 1)
        bar_length = int(score_percent / 5)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        print(f"  {medals[i]}  {rec['role']}")
        print(f"      Match Score : [{bar}] {score_percent}%")
        print(f"      Role Skills : {', '.join(rec['skills'][:5])}...")
        print()

    print("─" * 55)
    print("    Tip: The higher the %, the better the match!")
    print("─" * 55 + "\n")


# ─────────────────────────────────────────────────────────────
# STEP 7: COLD START HANDLER
# What if user provides skills not in our dataset?
# ─────────────────────────────────────────────────────────────

def handle_cold_start(user_skills, vocabulary):
    """
    Checks if user's skills exist in our vocabulary.
    Shows a warning for unknown skills.
    """
    cleaned = [s.strip().lower().replace(" ", "_") for s in user_skills]
    unknown = [s for s in cleaned if s not in vocabulary]
    known   = [s for s in cleaned if s in vocabulary]

    if unknown:
        print(f"\n    Unknown skills (not in database): {', '.join(unknown)}")
        print(f"  Recognized skills: {', '.join(known) if known else 'None'}")

    if not known:
        print("\n  None of your skills matched our database.")
        print("    Available skills include:")
        print("      " + ", ".join(sorted(vocabulary)[:20]) + "...")
        return False
    return True


# ─────────────────────────────────────────────────────────────
# MAIN PROGRAM — Runs when you execute this file
# ─────────────────────────────────────────────────────────────

def main():
    # Find and load the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "raw_skills.csv")

    try:
        dataset = load_dataset(csv_path)
    except FileNotFoundError:
        print(" Error: raw_skills.csv not found! Place it in the same folder.")
        return

    # Build vocabulary and IDF scores from the full dataset
    vocabulary = build_vocabulary(dataset)
    idf = compute_idf(dataset, vocabulary)

    display_banner()

    print("  Enter at least 3 of your skills.")
    print("  Examples: python, sql, aws, docker, javascript, react")
    print("  (Type 'quit' to exit)\n")

    while True:
        raw_input_str = input("  Enter your skills (comma-separated): ").strip()

        if raw_input_str.lower() == 'quit':
            print("\n   Thanks for using Tech Stack Recommender!\n")
            break

        # Split by comma
        user_skills = [s.strip() for s in raw_input_str.split(",") if s.strip()]

        # Validate: need at least 3 skills
        if len(user_skills) < 3:
            print("   Please enter at least 3 skills for accurate results.\n")
            continue

        # Cold start check
        if not handle_cold_start(user_skills, vocabulary):
            continue

        # Get top 3 recommendations
        results = recommend(user_skills, dataset, vocabulary, idf, top_n=3)

        # Display them nicely
        display_results(results, user_skills)

        # Ask if user wants to try again
        again = input("   Try with different skills? (yes/no): ").strip().lower()
        if again not in ['yes', 'y']:
            print("\n   Thanks for using Tech Stack Recommender!\n")
            break
        print()


if __name__ == "__main__":
    main()
