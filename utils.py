import json
import pickle
import os

def save_pickle(data, filepath):
    """Saves data to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved data to {filepath}")

def load_pickle(filepath):
    """Loads data from a pickle file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded data from {filepath}")
    return data

def save_json(data, filepath):
    """Saves data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {filepath}")

def load_json(filepath):
    """Loads data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded data from {filepath}")
    return data 