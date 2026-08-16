# Satellite Risk Index

Multi-Constellation Collision Risk Score Estimator — a spatial-temporal ML model estimating satellite conjunction risk in LEO.

## Live Demo
Experience the real-time simulation: **[https://PriyanshKotadia.github.io/satellite-risk-index/](https://PriyanshKotadia.github.io/satellite-risk-index/)**

> ![Globe Visualization](docs/screenshot.png)

*Visual Style: A deep navy/black space background is rendered using a highly performant static procedural canvas starfield (chosen over a CDN skybox texture to ensure perfect offline reliability and avoid broken links). Satellites orbit in real-time simulation along their orbital planes. Objects glow cyan/electric-blue for normal risk, shifting to an amber-to-red gradient for high-risk targets. A technical, glowing HUD overlay displays top SHAP drivers and risk metrics when a satellite is selected, while highlighting the entire parent constellation. Users can also search and filter satellites by name, constellation, or country.*

## How It Works
The project predicts satellite conjunction risk using a full end-to-end data pipeline:
1. **Data Prep**: Raw trajectory data is cleaned and enriched (calculating orbital shells).
2. **Feature Engineering**: Calculates spatial density and relative velocity metrics. (Note: Initial bugs with GEO density bias were corrected using shell-area normalization).
3. **LightGBM Model**: A tuned gradient boosting model predicts the risk score (R).
4. **Static Predictions**: Final predictions and SHAP (feature importance) values are pre-computed into a static `predictions.json` (6.5MB).
5. **deck.gl 3D Globe Frontend**: A WebGL-powered frontend simulates the orbits and visualizes the model's predictions.

For full technical details, see:
- [Methodology](docs/methodology.md)
- [Model Card](docs/model_card.md)
- [Data Dictionary](docs/data_dictionary.md)

## Credits
- **Earth Texture**: NASA Blue Marble dataset (Public Domain), via the `deck.gl-data` / `Three.js` repositories.
- **Space Texture**: Procedural canvas starfield to ensure 100% offline reliability without external CDN dependencies.

## Local Development
To run the full python pipeline and generate predictions:
```bash
pip install -r requirements.txt
python src/predict.py
```

To serve the frontend visualization locally:
```bash
python -m http.server 8000
```
Then navigate to `http://localhost:8000/web/` in your browser.

**Status: Complete & Deployed**
