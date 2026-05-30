# Land Cost Prediction - Supervised ML

A machine learning project that predicts land costs based on 20 years of historical data, served via a FastAPI REST API.

## Model

- **Algorithm**: XGBoost (via scikit-learn Pipeline)
- **Input features**: `region`, `land_size_sqft`, `distance_to_center_km`, `year`, `month`
- **Output**: Predicted land cost

## Project Structure

```
├── main.py                              # FastAPI app
├── land_cost_predictor_pipeline.joblib  # Trained model
├── PlotPricePrediction.ipynb            # Training notebook
└── .gitignore
```

## Setup

```bash
python -m pip install fastapi uvicorn joblib pandas scikit-learn==1.6.1 xgboost
```

## Run the API

```bash
python main.py
```

Server starts at `http://localhost:8000`

## API Endpoints

| Method | Endpoint   | Description          |
|--------|------------|----------------------|
| GET    | `/health`  | Check model status   |
| POST   | `/predict` | Get cost prediction  |
| GET    | `/docs`    | Interactive API UI   |

## Sample Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "region": "urban",
      "land_size_sqft": 1200,
      "distance_to_center_km": 5.2,
      "year": 2024,
      "month": 6
    }
  }'
```

## Sample Response

```json
{
  "predicted_cost": 4500000.0
}
```
