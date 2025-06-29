import json
import sys
from argparse import ArgumentParser
from typing import List
from time import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


try:
    from sklearn.metrics import classification_report
except ImportError:
    # sklearn is optional
    def classification_report(y, y_pred):
        print("sklearn is not installed, skipping classification report")

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_ingredients_df = pd.read_parquet("data/embed_ingredients_df.parquet")


def get_most_similar(ingredient, df):
    # Embed the ingredient
    query_vec = model.encode([ingredient.lower()], convert_to_numpy=True)[0]
    # Compare to each USDA food
    all_vecs = np.vstack(df["embedding"].values)
    # Normalize vectors
    query_norm = query_vec / np.linalg.norm(query_vec)
    all_norms = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    # Compute cosine similarities (dot products of normalized vectors)
    cosine_similarities = all_norms @ query_norm
    # Get index of max cosine similarity (best match)
    top_idx = np.argmax(cosine_similarities)
    closest_match = df.iloc[top_idx]
    similarity_score = cosine_similarities[top_idx]
    return closest_match, similarity_score


def is_ingredient_keto(ingredient: str) -> bool:
    match, dist = get_most_similar(ingredient, embedding_ingredients_df)
    # print(f"for ingredient: {ingredient}, match: {match['ingredient_name']}")
    if match["is_keto"]:
        return True  # Low-carb
    else:
        return False  # High-carb


def is_ingredient_vegan(ingredient: str) -> bool:
    match, dist = get_most_similar(ingredient, embedding_ingredients_df)
    # print(f"for ingredient: {ingredient}, match: {match['ingredient_name']}")
    if match["is_vegan"]:
        return True  # Vegan
    else:
        return False  # non Vegan


def to_list(lst):
    if type(lst) == str:
        return json.loads(lst)
    return lst


def is_keto(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_keto, to_list(ingredients)))


def is_vegan(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_vegan, to_list(ingredients)))

