# ML Backend Service

This is a Python FastAPI service that performs real machine learning processing.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
python main.py
```

The service will run on `http://localhost:8000`

## Deployment Options

### Option 1: Railway (Recommended - Easy)
1. Create account at https://railway.app
2. Create new project
3. Deploy from this folder
4. Railway will auto-detect and deploy the FastAPI app
5. Copy the deployment URL

### Option 2: Render
1. Create account at https://render.com
2. Create new Web Service
3. Connect your repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Copy the deployment URL

### Option 3: Google Cloud Run
1. Install gcloud CLI
2. Build and deploy:
```bash
gcloud run deploy ml-backend --source . --region us-central1
```

### Option 4: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## API Endpoints

### POST /process
Process ML analysis

**Request Body:**
```json
{
  "dataset": "Iris Dataset",
  "model": "Random Forest",
  "learning_type": "supervised",
  "visualizations": ["roc", "confusion_matrix"],
  "test_size": 0.2,
  "random_state": 42
}
```

**Response:**
```json
{
  "success": true,
  "results": {
    "accuracy": 0.942,
    "precision": 0.915,
    "recall": 0.898,
    "f1_score": 0.906,
    "confusion_matrix": [[85, 12], [8, 95]],
    "feature_importance": [...]
  }
}
```

## After Deployment

Once deployed, add the Python backend URL as a secret in Lovable:
1. The edge function will use this URL to call your Python backend
2. Add secret: `PYTHON_ML_BACKEND_URL` with your deployment URL
