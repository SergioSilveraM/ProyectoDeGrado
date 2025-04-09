# **Clasificador Híbrido para la Predicción del Nivel de Inglés en Saber Pro: Un Enfoque Basado en Ensamble**  

La evaluación del nivel de inglés en estudiantes universitarios colombianos, mediante las Pruebas Saber Pro, constituye una herramienta clave para diagnosticar competencias comunicativas en un contexto globalizado. Estas pruebas estandarizadas no solo permiten medir el dominio del idioma, sino que también ofrecen insumos valiosos para analizar el impacto de variables socioeconómicas, académicas y demográficas en el desempeño estudiantil.

En este estudio, se propone un modelo de aprendizaje automático (ML) orientado a la predicción del nivel de inglés alcanzado en la prueba **Saber Pro**, a partir de datos históricos recolectados entre $2018$ y $2022$. El enfoque se basa en un esquema de ensamble tipo **stacking**, combinando la capacidad de generalización de **XGBoost** y **Multi-Layer Perceptron (MLP)** como clasificadores base, con la robustez de un **Random Forest** como metamodelo final.

El desarrollo del trabajo se estructura en las siguientes etapas:

* **Análisis Exploratorio de Datos (EDA)**: Se caracterizan las variables predictoras, se identifican relaciones no lineales y se analizan distribuciones que podrían incidir en el rendimiento en inglés.
Entrenamiento y Comparación de Modelos Base: Se evalúan clasificadores individuales tradicionales y se contrastan sus métricas de desempeño frente a la arquitectura de ensamble propuesta.
* **Construcción del Modelo Apilado:** Se implementa un modelo híbrido mediante **stacking**, con validación cruzada y ajuste de hiperparámetros, optimizando métricas como accuracy, F1-score y AUC-ROC.
* **Benchmarking Final:** Se comparan los resultados del modelo híbrido con enfoques base, evidenciando mejoras sustanciales en la predicción del nivel de inglés.
El objetivo del informe es construir un modelo robusto y explicativo que permita anticipar el desempeño en inglés con alto grado de **precisión**, brindando apoyo a procesos de análisis educativo y toma de decisiones pedagógicas.

## **Integrantes**
- Sergio Silvera
- Alvaro Romano 
- Miguel Cabrera

## **Tabla de Contenido**
```{tableofcontents}
```
