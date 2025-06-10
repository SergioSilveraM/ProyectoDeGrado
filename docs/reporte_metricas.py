import pandas as pd
import os
from sklearn.metrics import classification_report

class ReporteMetricas:
    def __init__(self, output_dir: str = "./Metrics"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        pd.options.display.float_format = '{:.3f}'.format

    def save(self, metrics: dict, model_name: str):
        rows = []
        metric_cols = ["precision", "recall", "f1-score", "support"]

        for tipo in ["train", "val", "test"]:
            if tipo == "train" and "accuracy_train" in metrics:
                y_true = metrics.get("y_train_true", metrics["y_train_fold"])
                y_pred = metrics.get("y_train_pred")
                acc = metrics["accuracy_train"]
                log = metrics["log_loss_train"]
                auc = metrics["auc_train"]
            elif tipo == "val" and "accuracy_val" in metrics:
                y_true = metrics.get("y_val_true", metrics.get("y_train_fold"))
                y_pred = metrics.get("y_val_pred")
                acc = metrics["accuracy_val"]
                log = metrics["log_loss_val"]
                auc = metrics["auc_val"]
            elif tipo == "test" and "accuracy_test" in metrics:
                y_true = metrics.get("y_test_fold")
                y_pred = metrics.get("y_test_pred")
                acc = metrics["accuracy_test"]
                log = metrics["log_loss_test"]
                auc = metrics["auc_test"]
            else:
                continue

            if y_true is None or y_pred is None:
                continue

            report_dict = classification_report(y_true, y_pred, output_dict=True)
            report_dict.pop("accuracy", None)

            df = pd.DataFrame(report_dict).T.reset_index()
            df.rename(columns={"index": "Class"}, inplace=True)
            df["Model"] = model_name
            df["Type"] = tipo
            df["accuracy"] = "-"
            df["log_loss"] = "-"
            df["auc"] = "-"

            df = df[["Model", "Type", "Class"] + metric_cols + ["accuracy", "log_loss", "auc"]]

            for col in metric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(3)
            df[metric_cols] = df[metric_cols].fillna(0)

            global_row = pd.DataFrame([{
                "Model": model_name,
                "Type": tipo,
                "Class": "global",
                "precision": "-",
                "recall": "-",
                "f1-score": "-",
                "support": "-",
                "accuracy": round(acc, 3),
                "log_loss": round(log, 3),
                "auc": round(auc, 3)
            }])

            df = pd.concat([df, global_row], ignore_index=True)
            rows.append(df)

        if not rows:
            print(f"No metrics saved for {model_name}")
            return

        df_final = pd.concat(rows, ignore_index=True)

        # Save CSV
        csv_path = os.path.join(self.output_dir, f"Metrics_{model_name}.csv")
        df_final.to_csv(csv_path, index=False)

        # Save JSON
        json_path = os.path.join(self.output_dir, f"Metrics_{model_name}.json")
        df_final.to_json(json_path, orient="records", indent=4)

        print(f"\nReport for model '{model_name}' saved:")
        print(f"   → CSV: {csv_path}")
        print(f"   → JSON: {json_path}")

    def load(self):
        files = [f for f in os.listdir(self.output_dir) if f.endswith(".csv") and f.startswith("Metrics_")]
        if not files:
            print("No metric reports found.")
            return None

        all_reports = pd.concat([pd.read_csv(os.path.join(self.output_dir, f)) for f in files], ignore_index=True)
        print(f"\nLoaded {len(files)} report(s)")
        return all_reports
