# filename: ml_service.py
import os
import io
import base64
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure datasets folder exists
DATASETS_DIR = "./datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)

app = FastAPI(title="ML Backend Service")

# CORS (allow all origins for development; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Config / Constants
# -----------------------------
PREDEFINED_DATASETS = {
    # sklearn dataset loaders (converted to DataFrame)
    "iris": lambda: _sklearn_to_df(load_iris(), target_name="target"),
    "wine": lambda: _sklearn_to_df(load_wine(), target_name="target"),
    "breast_cancer": lambda: _sklearn_to_df(load_breast_cancer(), target_name="target"),
    "digits": lambda: _sklearn_to_df(load_digits(), target_name="target"),
    # placeholders for common CSV-based datasets; place CSVs under ./datasets/
    "diabetes": lambda: pd.read_csv(os.path.join(DATASETS_DIR, "diabetes.csv")),
    "titanic": lambda: pd.read_csv(os.path.join(DATASETS_DIR, "titanic.csv")),
    # Examples using OpenML may be heavy; included as wrappers (requires internet at runtime)
    "mnist": lambda: fetch_openml_to_df("mnist_784", target_col_name="target"),
    "winequality-red": lambda: fetch_openml_to_df("wine-quality-red", target_col_name="target"),
    # Example custom local CSVs
    "heart": lambda: pd.read_csv(os.path.join(DATASETS_DIR, "heart.csv")),
}

# available model shorthand -> class
AVAILABLE_MODELS = {
    "logisticregression": LogisticRegression,
    "logistic": LogisticRegression,
    "randomforest": RandomForestClassifier,
    "random-forest": RandomForestClassifier,
    "rf": RandomForestClassifier,
    "neuralnetwork": MLPClassifier,
    "neuralnet": MLPClassifier,
    "mlp": MLPClassifier,
    "svc": SVC,
    "svm": SVC,
}

# allowed visualization keys
AVAILABLE_VISUALIZATIONS = [
    "roc_curve",
    "confusion_matrix",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "heatmap",
    "plots",  # simple feature vs target plots
]

# -----------------------------
# Helpers
# -----------------------------
def _sklearn_to_df(ds, target_name="target"):
    """Convert sklearn Bunch dataset to DataFrame with target."""
    df = pd.DataFrame(ds.data, columns=[c for c in ds.feature_names])
    df[target_name] = ds.target
    return df

def fetch_openml_to_df(name: str, target_col_name: str = "target"):
    """Fetch OpenML dataset by name and return DataFrame.
    NOTE: requires internet at runtime. Caller should handle network issues."""
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name, version=1, as_frame=True)
    df = ds.frame.copy()
    # many OpenML datasets have different target column names; try common ones
    if "class" in df.columns:
        df.rename(columns={"class": target_col_name}, inplace=True)
    elif ds.target_names is not None:
        # try to rename the existing target column if possible
        # otherwise, ensure a column named target_col_name exists
        pass
    # if there's no explicit target column, try to set the last column as target
    if target_col_name not in df.columns:
        df[target_col_name] = df.iloc[:, -1]
    return df

def df_to_base64_image(plt_figure) -> str:
    """Take a matplotlib figure and return a PNG base64 string."""
    buf = io.BytesIO()
    plt_figure.savefig(buf, format="png", bbox_inches="tight")
    plt.close(plt_figure)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def normalize_model_name(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

def infer_target_col(df: pd.DataFrame, provided: Optional[str]) -> str:
    if provided and provided in df.columns:
        return provided
    if "target" in df.columns:
        return "target"
    # fall back to last column
    return df.columns[-1]

# -----------------------------
# Pydantic Request model
# -----------------------------
class MLRequest(BaseModel):
    dataset: str
    model: str
    learning_type: Optional[str] = "classification"  # we assume classification for now
    visualizations: List[str]
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42
    target_col: Optional[str] = None  # optional name of target column in dataset

# -----------------------------
# ML Processor
# -----------------------------
class MLProcessor:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        model_name: str = "randomforest",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.df = df.copy()
        self.target_col = infer_target_col(self.df, target_col)
        self.test_size = test_size
        self.random_state = random_state
        self.model_name_raw = model_name
        self.model_key = normalize_model_name(model_name)
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.model = None

    def preprocess(self):
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset")
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        # simple fillna for numeric columns (improvements possible)
        X = X.select_dtypes(include=[np.number]).fillna(0)
        self.feature_names = list(X.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=None
        )

    def build_model(self):
        ModelClass = AVAILABLE_MODELS.get(self.model_key)
        if ModelClass is None:
            raise ValueError(f"Model '{self.model_name_raw}' not supported. Supported: {list(AVAILABLE_MODELS.keys())}")
        # configure model defaults
        if ModelClass is SVC:
            # prefer probability estimates for ROC if requested (may slow training)
            self.model = ModelClass(probability=True, random_state=self.random_state)
        elif ModelClass is MLPClassifier:
            self.model = ModelClass(random_state=self.random_state, max_iter=500)
        elif ModelClass is RandomForestClassifier:
            self.model = ModelClass(random_state=self.random_state)
        else:
            # LogisticRegression or others
            try:
                self.model = ModelClass(max_iter=500, random_state=self.random_state)
            except TypeError:
                # some classes may not accept random_state
                self.model = ModelClass()

    def train(self):
        self.preprocess()
        self.build_model()
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(self.X_test)
                # for binary, take column 1
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    y_pred_proba = proba
                else:
                    y_pred_proba = proba
            except Exception:
                y_pred_proba = None

        num_classes = len(np.unique(self.y_test))
        avg = "binary" if num_classes == 2 else "macro"

        metrics = {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred, average=avg, zero_division=0)),
            "recall": float(recall_score(self.y_test, y_pred, average=avg, zero_division=0)),
            "f1_score": float(f1_score(self.y_test, y_pred, average=avg, zero_division=0)),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
        }

        viz_images = {}

        # ROC: if predict_proba available
        if y_pred_proba is not None:
            try:
                # multiclass: compute ROC per class (one-vs-rest)
                if num_classes == 2:
                    # need single column of positive-class probabilities
                    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
                        probs = y_pred_proba[:, 1]
                    else:
                        # if model returned shape (n_samples, 1) or (n_samples,), try flatten
                        probs = np.ravel(y_pred_proba)
                    fpr, tpr, _ = roc_curve(self.y_test, probs)
                    roc_auc = auc(fpr, tpr)
                    # plot
                    fig = plt.figure(figsize=(6, 5))
                    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                    plt.plot([0, 1], [0, 1], linestyle="--")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve")
                    plt.legend()
                    viz_images["roc_curve"] = df_to_base64_image(fig)
                    metrics["roc_auc"] = float(roc_auc)
                else:
                    # multilabel binarize y_test
                    classes = np.unique(self.y_test)
                    y_test_bin = label_binarize(self.y_test, classes=classes)
                    # if predict_proba provides one column per class
                    if y_pred_proba.shape[1] == len(classes):
                        fig = plt.figure(figsize=(6, 5))
                        aucs = []
                        for i, cls in enumerate(classes):
                            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                            roc_auc = auc(fpr, tpr)
                            aucs.append(roc_auc)
                            plt.plot(fpr, tpr, label=f"class {cls} (AUC={roc_auc:.2f})")
                        plt.plot([0, 1], [0, 1], linestyle="--")
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.title("ROC Curves (one-vs-rest)")
                        plt.legend(fontsize="small")
                        viz_images["roc_curve"] = df_to_base64_image(fig)
                        metrics["roc_auc_per_class"] = {str(c): float(a) for c, a in zip(classes, aucs)}
            except Exception:
                pass

        # Heatmap: correlation of numeric features
        try:
            corr = self.df.drop(columns=[self.target_col]).select_dtypes(include=[np.number]).corr()
            fig = plt.figure(figsize=(6, 5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Feature Correlation Heatmap")
            viz_images["heatmap"] = {
                "image_base64": df_to_base64_image(fig),
                "matrix": corr.values.tolist(),
                "features": corr.columns.tolist(),
            }
        except Exception:
            pass

        # Confusion matrix plot
        try:
            cm = confusion_matrix(self.y_test, y_pred)
            fig = plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            viz_images["confusion_matrix_image"] = df_to_base64_image(fig)
        except Exception:
            pass

        # Simple feature plots (scatter) - top up to 4 numeric features vs target for basic visualization
        try:
            numeric_cols = self.df.drop(columns=[self.target_col]).select_dtypes(include=[np.number]).columns
            plots = []
            for i, col in enumerate(numeric_cols[:4]):
                fig = plt.figure(figsize=(4, 3))
                plt.scatter(self.df[col], self.df[self.target_col], alpha=0.6)
                plt.xlabel(col)
                plt.ylabel(str(self.target_col))
                plt.title(f"{col} vs {self.target_col}")
                plots.append({"feature": col, "image_base64": df_to_base64_image(fig)})
            if plots:
                viz_images["plots"] = plots
        except Exception:
            pass

        # Feature importance or coefficients
        try:
            if hasattr(self.model, "feature_importances_"):
                fi = [
                    {"feature": f, "importance": float(i)}
                    for f, i in zip(self.feature_names, self.model.feature_importances_)
                ]
                metrics["feature_importance"] = fi
            elif hasattr(self.model, "coef_"):
                coefs = np.mean(np.abs(self.model.coef_), axis=0) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
                metrics["feature_importance"] = [
                    {"feature": f, "importance": float(i)} for f, i in zip(self.feature_names, coefs)
                ]
        except Exception:
            pass

        return {"metrics": metrics, "visualizations": viz_images}

# -----------------------------
# Dataset loader
# -----------------------------
def load_dataset_by_name(name: str) -> pd.DataFrame:
    name_norm = (name or "").strip().lower()
    # first check predefined map
    if name_norm in PREDEFINED_DATASETS:
        loader = PREDEFINED_DATASETS[name_norm]
        try:
            df = loader()
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Loader did not return a DataFrame")
            return df
        except FileNotFoundError as fe:
            raise fe
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load predefined dataset '{name}': {e}")
    # fallback: if a CSV with that name exists in datasets folder
    csv_path = os.path.join(DATASETS_DIR, f"{name_norm}.csv")
    if os.path.exists(csv_path):
        try:
            return safe_read_csv(csv_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read CSV '{csv_path}': {e}")
    raise ValueError(f"Dataset '{name}' not supported or not found. Available: {list(PREDEFINED_DATASETS.keys()) + ['<uploaded CSV names in datasets folder']}")


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": "ML Backend Service Running"}


@app.get("/datasets")
def list_datasets():
    # list predefined + CSV files in datasets dir
    csv_files = [f[:-4] for f in os.listdir(DATASETS_DIR) if f.lower().endswith(".csv")]
    return {"predefined": list(PREDEFINED_DATASETS.keys()), "uploaded_csvs": csv_files}


@app.get("/models")
def list_models():
    return {"models": list(AVAILABLE_MODELS.keys())}


@app.get("/visualizations")
def list_visualizations():
    return {"visualizations": AVAILABLE_VISUALIZATIONS}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    """Upload a CSV dataset and save it as ./datasets/<name>.csv (or original filename)"""
    try:
        contents = await file.read()
        filename = (name or os.path.splitext(file.filename)[0]).strip().lower() + ".csv"
        path = os.path.join(DATASETS_DIR, filename)
        with open(path, "wb") as f:
            f.write(contents)
        # sanity-check read
        df = pd.read_csv(path)
        return {"success": True, "filename": filename, "rows": len(df), "columns": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_ml(request: MLRequest):
    """
    Main ML processing endpoint.
    Expects:
    {
      "dataset": "<name>",
      "model": "<model name>",
      "learning_type": "classification",
      "visualizations": ["roc_curve","confusion_matrix","heatmap","plots"],
      "test_size": 0.2,
      "random_state": 42,
      "target_col": "target"  (optional)
    }
    """
    try:
        # Normalize requested visualization keys
        viz_keys = [v.strip().lower() for v in (request.visualizations or [])]
        # Load dataset (predefined or uploaded CSV)
        df = load_dataset_by_name(request.dataset)

        # Build and run processor
        processor = MLProcessor(
            df=df,
            target_col=request.target_col,
            model_name=request.model,
            test_size=request.test_size,
            random_state=request.random_state,
        )
        # train & evaluate
        processor.train()
        eval_results = processor.evaluate()

        # Filter metrics/visualizations per request
        metrics = eval_results.get("metrics", {})
        all_viz_images = eval_results.get("visualizations", {})

        filtered_results: Dict[str, Any] = {}
        # include requested scalar metrics
        for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
            if metric_name in viz_keys or metric_name in AVAILABLE_VISUALIZATIONS:
                # always include these if available
                if metric_name in metrics:
                    filtered_results[metric_name] = metrics[metric_name]

        # include ROC
        if any(k in ("roc", "roc_curve") for k in viz_keys) and "roc_curve" in all_viz_images:
            filtered_results["roc_curve_image_base64"] = all_viz_images["roc_curve"]
            if "roc_auc" in metrics:
                filtered_results["roc_auc"] = metrics.get("roc_auc")
            if "roc_auc_per_class" in metrics:
                filtered_results["roc_auc_per_class"] = metrics.get("roc_auc_per_class")

        # confusion
        if any(k in ("confusion", "confusion_matrix") for k in viz_keys):
            if "confusion_matrix" in metrics:
                filtered_results["confusion_matrix"] = metrics["confusion_matrix"]
            if "confusion_matrix_image" in all_viz_images:
                filtered_results["confusion_matrix_image_base64"] = all_viz_images["confusion_matrix_image"]

        # feature importance
        if any(k in ("feature", "feature_importance") for k in viz_keys) and "feature_importance" in metrics:
            filtered_results["feature_importance"] = metrics["feature_importance"]

        # heatmap
        if any(k in ("heatmap", "correlation", "feature_heatmap") for k in viz_keys) and "heatmap" in all_viz_images:
            filtered_results["heatmap"] = all_viz_images["heatmap"]

        # plots
        if any(k in ("plots", "scatter", "feature_plots") for k in viz_keys) and "plots" in all_viz_images:
            filtered_results["plots"] = all_viz_images["plots"]

        # include model metadata
        filtered_results["model_used"] = request.model
        filtered_results["dataset_used"] = request.dataset
        filtered_results["target_column"] = processor.target_col

        return {"success": True, "results": filtered_results}

    except FileNotFoundError as fe:
        # helpful error for missing local CSVs
        raise HTTPException(status_code=404, detail=str(fe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
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
