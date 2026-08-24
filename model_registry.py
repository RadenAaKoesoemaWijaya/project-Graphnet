import os
import json
import shutil
from datetime import datetime
import glob
import streamlit as st

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
REGISTRY_META = "registry_meta.json"

def init_registry():
    """Ensure models directory and registry meta file exist."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    meta_path = os.path.join(MODELS_DIR, REGISTRY_META)
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump({"versions": []}, f)

def get_versions():
    """Get list of saved model versions."""
    init_registry()
    meta_path = os.path.join(MODELS_DIR, REGISTRY_META)
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
            return meta.get("versions", [])
    except json.JSONDecodeError:
        return []

def save_model_version(temp_prefix, metrics=None):
    """Save the models from a temp prefix to a versioned directory."""
    init_registry()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"model_v_{timestamp}"
    version_dir = os.path.join(MODELS_DIR, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    # Move files from temp_prefix to version_dir
    files_to_move = glob.glob(f"{temp_prefix}*")
    saved_files = []
    for src in files_to_move:
        if os.path.isfile(src):
            filename = os.path.basename(src).replace("fraud_detector", version_name)
            dst = os.path.join(version_dir, filename)
            shutil.copy2(src, dst)
            saved_files.append(filename)
            
    # Update registry meta
    versions = get_versions()
    versions.append({
        "version": version_name,
        "created_at": timestamp,
        "metrics": metrics or {},
        "files": saved_files
    })
    
    meta_path = os.path.join(MODELS_DIR, REGISTRY_META)
    with open(meta_path, "w") as f:
        json.dump({"versions": versions}, f, indent=4)
        
    return version_name

def load_model_version(version_name, dest_prefix):
    """Load a specific version back into the dest_prefix so the app can use it."""
    version_dir = os.path.join(MODELS_DIR, version_name)
    if not os.path.exists(version_dir):
        return False
        
    # Copy files back to dest_prefix
    for filename in os.listdir(version_dir):
        src = os.path.join(version_dir, filename)
        if os.path.isfile(src):
            new_name = filename.replace(version_name, os.path.basename(dest_prefix))
            dst = os.path.join(os.path.dirname(dest_prefix), new_name)
            shutil.copy2(src, dst)
    return True
