# ***Proyecto de Segmentación de Clientes***

## Descripción General

Este proyecto desarrolla un sistema experto de ***segmentación de clientes morosos*** mediante la integración de técnicas de ***machine learning*** y un sistema de ***inferencia difusa***. El objetivo final es clasificar a los clientes en segmentos de riesgo que permitan ***priorizar recursos de cobranza*** y diseñar estrategias más efectivas, considerando tanto datos objetivos (variables extraidas a través de FI) como criterios subjetivos derivados del juicio experto.

## Objetivos

1. ***Construir un sistema de segmentación híbrido:*** Integrar modelos explicativos de ***machine learning*** con un sistema difuso basado en reglas para segmentar clientes según su perfil de riesgo y probabilidad de cumplimiento.

2. ***Capturar incertidumbre y subjetividad:*** Modelar con lógica difusa variables complejas como la disposición de pago, el nivel de contacto, la franja de mora o la antigüedad de la obligación, que no pueden ser descritas completamente por reglas rígidas.

3. ***Diseñar reglas lingüísticas de segmentación:*** Definir un conjunto de reglas basadas en conocimiento experto y variables modeladas (incluyendo las salidas del modelo ML) para agrupar clientes en segmentos operativos accionables.

4. ***Facilitar decisiones operativas:*** Ofrecer un sistema consultable que permita a los agentes de cobranza identificar con rapidez el tipo de cliente y adaptar sus estrategias de gestión.

## Fases del Proyecto

### Fase 1: Modelado - Machine Learning
- Consolidación de datos históricos y de gestión
- Entrenamiento de modelo explicativo (XGBoost)
- Obtención de probabilidades por clase como insumo para el sistema difuso

### Fase 2: Modelado - Lógica Difusa
- Caracterización de variables clave
- Definición de funciones de pertenencia
- Creación de reglas de inferencia basadas en juicio experto
- Validación conceptual y funcional del sistema

### Fase 3: Integración y Despliegue
- Integración del modelo ML con el sistema difuso
- Desarrollo de una interfaz de consulta
- Ejecución de prueba piloto con datos reales


## Tabla de Contenido

```{tableofcontents}
```
