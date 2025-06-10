# model_evaluator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score, log_loss, roc_auc_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import lime.lime_tabular

class ModelEvaluator:
    """
    Encapsulates evaluation, visualization, and explainability utilities for classification models.
    """

    def __init__(self, class_labels):
        """
        class_labels: list or array of class labels (e.g., [0, 1, 2])
        """
        self.class_labels = class_labels

    def evaluate_model(self, model, X_data, y_data):
        """
        Evaluate a trained classification model on a dataset.
        Returns metrics, classification report, confusion matrix, and predictions.
        """
        y_pred = model.predict(X_data)
        y_proba = model.predict_proba(X_data)
        f1 = f1_score(y_data, y_pred, average='weighted')
        acc = accuracy_score(y_data, y_pred)
        loss = log_loss(y_data, y_proba)
        auc_score = roc_auc_score(y_data, y_proba, multi_class='ovr', average='weighted')
        report = classification_report(y_data, y_pred)
        cm = confusion_matrix(y_data, y_pred)
        return f1, acc, loss, auc_score, report, cm, y_proba, y_pred

    def save_metrics_folds(self, folds_metrics: list, filename: str) -> pd.DataFrame:
        """
        Save per-fold metrics and summary statistics (mean, std) to CSV.
        """
        df = pd.DataFrame(folds_metrics)
        metric_cols = df.columns.drop('fold') if 'fold' in df.columns else df.columns
        mean_row = df[metric_cols].mean().to_dict()
        std_row = df[metric_cols].std().to_dict()
        mean_row['fold'] = 'mean'
        std_row['fold'] = 'std'
        df_final = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
        df_final.to_csv(filename, index=False)
        print(f"\nFold metrics + summary saved to: {filename}")
        return df_final

    def plot_confusion_matrix(self, cm, title):
        """
        Plot confusion matrix using provided confusion matrix and labels.
        """
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_labels)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(title)
        plt.show()

    def plot_roc_multiclass(self, y_true, y_proba, title="AUC-ROC Curve (Multiclass)"):
        """
        Plot multiclass ROC curve for predicted probabilities.
        """
        y_bin = label_binarize(y_true, classes=self.class_labels)
        n_classes = len(self.class_labels)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i],
                     label=f"Class {self.class_labels[i]} (AUC = {roc_auc[i]:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def get_lime_explainer(self, pipeline, X_train_raw, y_train_raw):
        """
        Build a LIME explainer from a scikit-learn pipeline.
        """
        preprocessor = pipeline.named_steps['preprocessor']
        X_transformed = preprocessor.transform(X_train_raw)
        feature_names = preprocessor.get_feature_names_out()
        class_names = np.unique(y_train_raw).astype(str)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_transformed,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )
        return explainer, X_transformed
