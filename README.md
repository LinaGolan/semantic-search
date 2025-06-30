# Semantic Ingredient Classifier

This is my solution to the search_by_ingredient challenge focused on classifying recipe ingredients as **keto** and **vegan**.

---

## 🧠 Solution Overview

The solution is divided into two main parts:

### 1. Dataset Creation
To build a reliable and semantically rich dataset of ingredients:

- **Extracting Ingredients:** I parsed all ingredients from the OpenSearch database, and used an LLM (GPT) to extract **only the ingredient names**, removing any measurements or quantities.
- **Labeling Ingredients:** For each unique ingredient, I queried GPT (with web search enabled) to determine:
  - ✅ Is it keto?
  - 🌱 Is it vegan?

- **Embedding:** Once labeled, each ingredient was transformed into a vector using an **embedding model**, to allow for semantic similarity searches.

This process resulted in a high-quality, labeled, and embedded ingredient dataset.

---

### 2. Semantic Search Classification

When classifying new ingredients from a recipe:

1. Each new ingredient is embedded using the same embedding model.
2. A **cosine similarity** search is performed against the labeled dataset.
3. The closest match is selected, and its `is_keto` and `is_vegan` values are returned as the classification result.

This approach supports fuzzy matching, even for non-identical or slightly varied ingredient names.

---

## ⚙️ Technologies Used

- **Python**
- **OpenSearch** (for original ingredient data)
- **OpenAI GPT** (for ingredient parsing and web-aided dietary classification)
- **Embedding Model** (e.g., `sentence-transformers`)
- **Cosine Similarity** (for semantic search)

---

## 🚀 Improvements & Future Work

With more time and resources, I would explore the following enhancements to improve accuracy, performance, and dataset quality:

- **Augment Dataset from External Sources:** Enrich the ingredient set by crawling structured data from external sources or leveraging curated datasets from platforms like [Kaggle](https://www.kaggle.com/) or USDA FoodData Central.
- **Benchmark Multiple Embedding Models:** Test alternative embedding models such as `OpenAI`, `SentenceTransformers`, `Cohere`, or `Instructor-XL` to determine which best captures ingredient semantics.
- **Cross-Validate Using LLMs:** Use multiple phrased prompts or ensemble-style querying with LLMs to reduce hallucinations and improve labeling reliability.
- **Add Manual or Heuristic Checks:** Incorporate simple rules (e.g., if the ingredient contains "bacon", it's not vegan) to catch common classification mistakes.
- **Improve Classification Confidence:** Use similarity thresholds or add confidence scores to indicate uncertain matches.
- **Expand Dietary Labels:** Extend support to additional dietary categories such as `gluten-free`, `low-carb`, and `paleo`.
- **Optimize and Scale the Pipeline:** Refactor the system into a modular, scalable pipeline capable of handling large datasets efficiently, with async batching and better resource utilization.