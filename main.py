import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.datasets import load_iris

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request Model
# -----------------------------
class MLRequest(BaseModel):
    dataset: str
    model: str
    learning_type: str
    visualizations: List[str]
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42

# -----------------------------
# ML Processor
# -----------------------------
class MLProcessor:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        model_name: str = "Random Forest",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        # Normalize model names for flexibility
        self.model_name = self.normalize_model_name(model_name)

        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

        self.available_models = {
            "logisticregression": LogisticRegression,
            "randomforest": RandomForestClassifier,
            "neuralnetwork": MLPClassifier,
        }

    def normalize_model_name(self, name: str) -> str:
        # remove spaces, dashes, underscores, lowercase
        return "".join(name.lower().split()).replace("-", "").replace("_", "")

    def preprocess(self):
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset")
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def train_and_evaluate(self):
        self.preprocess()
        ModelClass = self.available_models.get(self.model_name)
        if ModelClass is None:
            raise ValueError(f"Model '{self.model_name}' not supported")

        # Configure model parameters
        if self.model_name == "neuralnetwork":
            model = ModelClass(random_state=self.random_state, max_iter=500)
        elif self.model_name == "randomforest":
            model = ModelClass(random_state=self.random_state)
        else:  # logisticregression
            model = ModelClass(max_iter=500, random_state=self.random_state)

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        # Predict probabilities if available
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(self.X_test)
            if proba.ndim == 2 and proba.shape[1] == 2:
                y_pred_proba = proba[:, 1]

        result = {}
        num_classes = len(np.unique(self.y_test))
        avg = "binary" if num_classes == 2 else "macro"

        result["accuracy"] = float(accuracy_score(self.y_test, y_pred))
        result["precision"] = float(precision_score(self.y_test, y_pred, average=avg, zero_division=0))
        result["recall"] = float(recall_score(self.y_test, y_pred, average=avg, zero_division=0))
        result["f1_score"] = float(f1_score(self.y_test, y_pred, average=avg, zero_division=0))

        # ROC curve for binary classification
        if y_pred_proba is not None and num_classes == 2:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            result["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}

        result["confusion_matrix"] = confusion_matrix(self.y_test, y_pred).tolist()

        # Feature importance / coefficients
        if hasattr(model, "feature_importances_"):
            result["feature_importance"] = [
                {"feature": f, "importance": float(i)}
                for f, i in zip(self.X_train.columns, model.feature_importances_)
            ]
        elif hasattr(model, "coef_"):
            coefs = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            result["feature_importance"] = [
                {"feature": f, "importance": float(i)}
                for f, i in zip(self.X_train.columns, coefs)
            ]

        # Correlation heatmap
        try:
            corr = self.df.drop(columns=[self.target_col]).corr()
            result["heatmap"] = corr.values.tolist()
            result["heatmap_features"] = corr.columns.tolist()
        except Exception:
            pass

        return result

# -----------------------------
# Load datasets
# -----------------------------
def load_dataset(name: str) -> pd.DataFrame:
    name_norm = (name or "").strip().lower()
    if name_norm in ("iris", "iris dataset", "iris_dataset"):
        ds = load_iris()
        df = pd.DataFrame(ds.data, columns=[c.replace(" (cm)", "").strip() for c in ds.feature_names])
        df["target"] = ds.target
        return df
    elif name_norm in ("diabetes", "diabetes.csv"):
        df = pd.read_csv("./datasets/diabetes.csv")
        return df
    elif name_norm in ("titanic", "titanic.csv"):
        df = pd.read_csv("./datasets/titanic.csv")
        return df
    else:
        raise ValueError(f"Dataset '{name}' not supported")

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": "ML Backend Service Running"}

@app.post("/process")
async def process_ml(request: MLRequest):
    try:
        viz_keys = [v.strip().lower() for v in (request.visualizations or [])]
        df = load_dataset(request.dataset)
        processor = MLProcessor(
            df,
            target_col="target",
            model_name=request.model,
            test_size=request.test_size,
            random_state=request.random_state,
        )
        results = processor.train_and_evaluate()

        filtered_results = {}
        for key in ["accuracy", "precision", "recall", "f1_score"]:
            filtered_results[key] = results.get(key)

        if any(k in ("roc", "roc_curve") for k in viz_keys) and "roc_curve" in results:
            filtered_results["roc_curve"] = results["roc_curve"]

        if any(k in ("confusion", "confusion_matrix") for k in viz_keys) and "confusion_matrix" in results:
            filtered_results["confusion_matrix"] = results["confusion_matrix"]

        if any(k in ("feature", "feature_importance") for k in viz_keys) and "feature_importance" in results:
            filtered_results["feature_importance"] = results["feature_importance"]

        if any(k in ("heatmap", "correlation", "feature_heatmap") for k in viz_keys) and "heatmap" in results:
            filtered_results["heatmap"] = {
                "matrix": results["heatmap"],
                "features": results.get("heatmap_features", []),
            }

        return {"success": True, "results": filtered_results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Run locally
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
