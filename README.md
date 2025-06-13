# 🧠 ***Optimización de la Gestión de Cobranza en una BPO Colombiana: Integración de Modelos de Aprendizaje Automático e Inferencia Difusa para la Segmentación y Priorización de Clientes***

Este proyecto propone el desarrollo de un sistema experto para la ***segmentación de clientes morosos***, integrando técnicas de **aprendizaje automático** con un sistema de **inferencia difusa**. El objetivo principal es optimizar la gestión de cobranzas mediante una clasificación de riesgo que permita priorizar recursos de forma más eficiente.

---

## 📌 Descripción técnica

La metodología se dividió en dos fases:

1. **Selección del modelo de clasificación**: se compararon algoritmos como Árboles de Decisión, Random Forest, Máquinas de Soporte Vectorial y Gradient Boosting, usando validación cruzada anidada.
2. **Integración con lógica difusa**: se construyó un sistema de inferencia que permite manejar la incertidumbre inherente a los procesos de cobranza.

El modelo seleccionado fue **XGBoost**, con métricas destacadas:

- Precisión: `0.844`  
- F1-Score: `0.825`  
- AUC: `0.96`  

La lógica difusa permitió segmentar a los clientes en cinco niveles de riesgo, mejorando la asignación operativa de esfuerzos de cobranza.

---

## 📂 Estructura del proyecto

```bash
ProyectoDeGrado/
├── Pruebas_anteriores/       # Backups o pruebas anteriores
├── Scripts/                  # Scripts en Python del sistema
├── docs/                     # Documentación, notebook y exportables
├── _images/                  # Recursos visuales usados en el proyecto


Continuar Editando...
