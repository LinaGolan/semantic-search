import json
import sys
from argparse import ArgumentParser
from typing import List
from time import time
import pandas as pd
import ast
import regex as re
from sentence_transformers import SentenceTransformer
import numpy as np

try:
    from sklearn.metrics import classification_report
except ImportError:
    # sklearn is optional
    def classification_report(y, y_pred):
        print("sklearn is not installed, skipping classification report")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Load ingredient embeddings df
embedding_ingredients_df = pd.read_parquet("data/embed_ingredients_df.parquet")

def get_most_similar(ingredient, df):
    """
    Find the most semantically similar ingredient from a dataframe of embeddings.

    Args:
        ingredient (str): Ingredient name to compare.
        df (pd.DataFrame): DataFrame containing 'embedding' column with vector embeddings of ingredients.

    Returns:
        The row from df with the highest cosine similarity.
    """
    # Generate embedding for the input ingredient
    query_vec = model.encode([ingredient.lower()], convert_to_numpy=True)[0]

    # Stack all existing embeddings into a single NumPy array
    all_vecs = np.vstack(df["embedding"].values)

    # Normalize vectors for cosine similarity
    query_norm = query_vec / np.linalg.norm(query_vec)
    all_norms = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)

    # Compute cosine similarities (dot products of normalized vectors)
    cosine_similarities = all_norms @ query_norm

    # Find the index of the most similar item
    top_idx = np.argmax(cosine_similarities)
    closest_match = df.iloc[top_idx]

    return closest_match

    
def is_ingredient_keto(ingredient: str) -> bool:
    match = get_most_similar(ingredient, embedding_ingredients_df)
    if match["is_keto"]:
        return True # Low-carb
    else:
        return False # High-carb


def is_ingredient_vegan(ingredient: str) -> bool:
    match = get_most_similar(ingredient, embedding_ingredients_df)
    if match["is_vegan"]:
        return True # Vegan
    else:
        return False # non Vegan
        
def to_list(lst):
    """
    If the input is a JSON-formatted string (e.g., '["item1", "item2"]'),
    it will be parsed into a list.
    """
    lst = re.findall(r"'([^']*)'", lst)
    return lst


def is_keto(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_keto, to_list(ingredients)))


def is_vegan(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_vegan, to_list(ingredients)))


def main(args):
    ground_truth = pd.read_csv(args.ground_truth, index_col=None)
    try:
        start_time = time()
        ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
        ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(
            is_vegan)

        end_time = time()
    except Exception as e:
        print(f"Error: {e}")
        return -1

    print("===Keto===")
    print(classification_report(
        ground_truth['keto'], ground_truth['keto_pred']))
    print("===Vegan===")
    print(classification_report(
        ground_truth['vegan'], ground_truth['vegan_pred']))
    print(f"== Time taken: {end_time - start_time} seconds ==")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth", type=str,
                        default="/usr/src/data/ground_truth_sample.csv")
    sys.exit(main(parser.parse_args()))