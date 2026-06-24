# 🎯 AI Recommendation System using TF-IDF & Cosine Similarity

An Artificial Intelligence project developed using Python that generates personalized recommendations by analyzing user preferences and calculating similarity scores.

This system converts user interests into numerical representations and identifies the most relevant recommendations using vector-based similarity techniques.

---

# 📌 Table of Contents

1. Project Overview
2. Problem Statement
3. Objectives
4. Features
5. Recommendation Architecture
6. Technologies Used
7. Dataset Structure
8. AI Concepts
9. System Workflow
10. Implementation Details
11. Similarity Logic
12. Test Cases
13. Results
14. Installation
15. Project Structure
16. Run Project
17. Future Improvements
18. Learning Outcomes
19. Author

---

# 📖 Project Overview

Recommendation systems are one of the most widely used applications of Artificial Intelligence.

Platforms recommend jobs, movies, products, and content by understanding user behavior and identifying similarity patterns.

This project simulates that process by recommending career roles based on user skills.

---

# 🎯 Problem Statement

Build a recommendation engine that:

✔ Accepts user preferences  
✔ Understands relationships between inputs  
✔ Calculates similarity scores  
✔ Generates personalized recommendations  
✔ Displays most relevant results  

---

# 🚀 Objectives

This project was developed to:

- Understand recommendation system fundamentals
- Apply vectorization techniques
- Learn similarity measurement
- Build personalized recommendation logic
- Understand real-world AI workflows

---

# ✨ Features

| Feature | Description |
|---------|-------------|
| User Input | Accept user skills |
| Skill Matching | Compare with dataset |
| TF-IDF Vectorization | Convert text into numbers |
| Cosine Similarity | Calculate closeness |
| Top-N Filtering | Return best results |
| Recommendation Output | Generate suggestions |

---

# 🏗 Recommendation Architecture

User Input  
↓  
Text Cleaning  
↓  
Vocabulary Generation  
↓  
TF Calculation  
↓  
IDF Calculation  
↓  
TF-IDF Transformation  
↓  
Cosine Similarity  
↓  
Ranking  
↓  
Top Recommendations

---

# 🛠 Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python | Development |
| Pandas | Data Processing |
| Scikit-learn | TF-IDF & Similarity |
| NumPy | Numerical Operations |

---

# 📂 Dataset Structure

| Column | Description |
|--------|-------------|
| Role | Job Category |
| Skills | Required Skills |

Example:

| Role | Skills |
|------|-------|
| Data Scientist | python sql machine_learning |
| DevOps Engineer | aws docker kubernetes |
| Web Developer | javascript react nodejs |

---

# 🧠 AI Concepts Implemented

## Vocabulary

A complete collection of unique skills available in the dataset.

Example:

python  
sql  
docker  
react  

---

## Term Frequency (TF)

Measures how important a skill is inside a role.

Formula:

TF = Frequency of Term / Total Terms

Example:

Python appears 2 times out of 8

TF = 0.25

---

## Inverse Document Frequency (IDF)

Measures how unique a skill is across all roles.

Formula:

IDF = log(Total Documents / Documents Containing Word)

Meaning:

Common skills receive lower weight.

---

## TF-IDF

Combines importance and uniqueness.

Formula:

TF × IDF

Result:

Transforms text into machine-readable vectors.

---

## Cosine Similarity

Measures similarity between user input and available roles.

Formula:

Similarity = cos(angle)

Interpretation:

0 → No Match

1 → Perfect Match

---

# ⚙ System Workflow

## Step 1 — Receive User Input

Example:

python, sql, machine_learning

↓

## Step 2 — Convert to TF-IDF

↓

## Step 3 — Calculate Similarity

↓

## Step 4 — Rank Results

↓

## Step 5 — Display Top Recommendations

---

# 🔍 Implementation Details

### Data Preparation

- Load dataset
- Clean text
- Normalize values

### Feature Engineering

- Build vocabulary
- Generate vectors

### Recommendation Engine

- Calculate similarity
- Filter Top Results

---

# 🧪 Test Cases

## Test 1

Input:

python, sql, machine_learning

Output:

Data Scientist ✅

---

## Test 2

Input:

aws, docker, kubernetes

Output:

DevOps Engineer ✅

---

## Test 3

Input:

javascript, react, nodejs

Output:

Web Developer / Full Stack ✅

---

## Test 4

Input:

java, kotlin, flutter

Output:

Mobile Developer ✅

---

# 📊 Results

| Input Type | Recommendation Quality |
|-----------|-----------------------|
| Exact Skills | Excellent |
| Partial Skills | Good |
| Mixed Skills | Moderate |

---

# 📈 Performance Highlights

✔ Personalized Results  
✔ Fast Similarity Search  
✔ Logical Matching  
✔ Accurate Recommendation Ranking  

---

# 📂 Project Structure

ai-recommendation-system/

│

├── dataset.csv  
├── recommendation.py  
├── outputs/  
├── screenshots/  
├── README.md  

---

# 💻 Installation

Clone Repository

git clone YOUR_REPOSITORY_LINK

Move Directory

cd ai-recommendation-system

Install Dependencies

pip install pandas

pip install numpy

pip install scikit-learn

---

# ▶ Run Project

python recommendation.py

---

# 📷 Example Output

Enter Skills:

python, sql, machine_learning

Top Recommendation:

Data Scientist

Similarity Score:

0.94

---

# 🔮 Future Improvements

- Add GUI
- Add Web Interface
- Real-Time Recommendation
- Deep Learning Integration
- Larger Dataset Support

---

# 📚 Learning Outcomes

✔ Recommendation Systems  
✔ TF-IDF Vectorization  
✔ Cosine Similarity  
✔ Feature Engineering  
✔ Pattern Matching  
✔ AI Decision Logic  

---

# 👩‍💻 Author

Minal Sadiq

BS Data Science Student  
Python | AI | Machine Learning | Data Science

---

⭐ If you found this project useful, consider giving it a star.
