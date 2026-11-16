import json
import pathlib
import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_project_data(data_dir="data"):
    """Loads all core datasets safely with caching and path handling."""
    path = pathlib.Path(data_dir)
    
    main_df = pd.read_csv(path / "main.csv")
    feedback_df = pd.read_csv(path / "feedback.csv")
    
    meta_file = path / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    
    images_path = path / "images.npz"
    images = np.load(images_path, allow_pickle=False) if images_path.exists() else None
    
    return main_df, feedback_df, meta, images