import os
import json
import pandas as pd
from src.data_prep import clean_catalog

def test_clean_catalog_smoke():
    """Smoke test to ensure clean_catalog runs without error on a mock DataFrame."""
    mock_data = pd.DataFrame({
        'norad_id': [1, 2, 3],
        'satellite_constellation': ['STARLINK', 'ONEWEB', None],
        'epoch': ['2023-01-01', '2023-01-01', '2023-01-01'],
        'mean_motion': [15.0, 14.0, 1.0],
        'eccentricity': [0.001, 0.002, 0.5],
        'inclination': [53.0, 87.0, 10.0],
        'altitude_km': [550.0, 1200.0, 35000.0],
        'ra_of_asc_node': [0.0, 0.0, 0.0],
        'arg_of_pericenter': [0.0, 0.0, 0.0],
        'mean_anomaly': [0.0, 0.0, 0.0],
        'ephemeris_type': [0, 0, 0],
        'classification_type': ['U', 'U', 'U'],
        'decay_date': [None, None, '2023-01-02']
    })
    
    cleaned = clean_catalog(mock_data)
    
    assert not cleaned.empty
    assert 'altitude_km' in cleaned.columns

def test_predictions_json_validity():
    """Smoke test to ensure predictions.json is valid and has correct structure."""
    path = os.path.join(os.path.dirname(__file__), '..', 'web', 'assets', 'predictions.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            
        assert isinstance(data, list)
        if len(data) > 0:
            first = data[0]
            required_keys = ['norad_id', 'predicted_R', 'top_3_shap_features', 'altitude_km']
            for k in required_keys:
                assert k in first, f"Missing {k} in predictions.json"
