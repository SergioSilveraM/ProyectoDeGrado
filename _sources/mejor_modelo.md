# **Mejor Modelo**
De los cuatro modelos basados en árboles probados durante el desarrollo de esta fase del proyecto, el modelo XGBoost presentó el mejor desempeño. A continuación, se detallan las métricas obtenidas en el conjunto de prueba:

    Test Metrics:
    F1-Score (Test): 0.8864
                precision    recall  f1-score   support

            0       0.92      0.97      0.95      3212
            1       0.81      0.73      0.77       969
            2       0.81      0.72      0.76       586

    accuracy                            0.89      4767
    macro avg       0.85      0.81      0.82      4767
    weighted avg    0.89      0.89      0.89      4767


![alt text](./img/image-1.png)

```python
from reporte_metricas.py import cargar_reportes

display(cargar_reportes("./Metrics/Test_Metrics"))

