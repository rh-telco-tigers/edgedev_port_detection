# prediction_app/utils.py
import json
import numpy as np

def get_categories_from_json(json_file):
    """
    Extract category names from a COCO format JSON file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    categories = {category['id']: category['name'] for category in data['categories']}
    sorted_categories = [categories[cat_id] for cat_id in sorted(categories)]
    
    return sorted_categories

def select_top_predictions_per_group(predictions, categories, n=8):
    """
    Select the top prediction for each port/cable type, ensuring diverse category representation.
    """
    predictions = np.array(predictions)
    if predictions.ndim > 1:
        predictions = predictions.squeeze()
    top_predictions_per_group = {}
    
    for score, category in sorted(zip(predictions, categories), reverse=True):
        base_name = "_".join(category.split("_")[:-1])
        if base_name not in top_predictions_per_group:
            top_predictions_per_group[base_name] = category
        if len(top_predictions_per_group) >= n:
            break
    
    selected_categories = list(top_predictions_per_group.values())
    return selected_categories
