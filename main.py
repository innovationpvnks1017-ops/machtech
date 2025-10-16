import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MLRequest(BaseModel):
    dataset: str
    model: str
    learning_type: str
    visualizations: List[str]
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42

class MLProcessor:
    def __init__(self, dataset_df: pd.DataFrame, target_col: str = "target",
                 model_name: str = "Random Forest",
                 test_size: float = 0.2, random_state: int = 42):
        self.df = dataset_df.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.available_models = {
            "Random Forest": RandomForestClassifier,
            "Logistic Regression": LogisticRegression,
            "SVM": SVC
        }
        self.model_name = model_name
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def preprocess(self):
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found")
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def train_and_evaluate(self):
        self.preprocess()
        
        ModelClass = self.available_models[self.model_name]
        if self.model_name == "SVM":
            model = ModelClass(probability=True, random_state=self.random_state)
        else:
            model = ModelClass(random_state=self.random_state)

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        # Basic metrics
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        result = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
        }

        # ROC / AUC
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            result["roc"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc),
            }

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        result["confusion_matrix"] = cm.tolist()

        # Feature importance
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            features = list(self.X_train.columns)
            result["feature_importance"] = [
                {"feature": f, "importance": float(i)} 
                for f, i in zip(features, imp)
            ]

        return result

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load preset dataset by name"""
    dataset_map = {
        "Iris Dataset": load_iris,
        "Wine Dataset": load_wine,
        "Breast Cancer Dataset": load_breast_cancer,
        "Diabetes Dataset": load_diabetes
    }
    
    if dataset_name not in dataset_map:
        dataset_name = "Iris Dataset"
    
    ds = dataset_map[dataset_name]()
    df = pd.DataFrame(ds.data, columns=ds.feature_names)
    df['target'] = ds.target
    return df

@app.get("/")
def read_root():
    return {"status": "ML Backend Service Running"}

@app.post("/process")
async def process_ml(request: MLRequest):
    try:
        # Load dataset
        df = load_dataset(request.dataset)
        
        # Process ML
        processor = MLProcessor(
            df, 
            target_col="target",
            model_name=request.model,
            test_size=request.test_size,
            random_state=request.random_state
        )
        
        results = processor.train_and_evaluate()
        
        return {
            "success": True,
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
