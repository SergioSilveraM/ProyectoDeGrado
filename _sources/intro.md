# ***Customer Segmentation Project***

## General Description

This project develops an expert system for the ***segmentation of delinquent clients*** by integrating ***machine learning*** techniques with a ***fuzzy inference system***. The ultimate goal is to classify clients into risk segments that allow for ***prioritizing collection resources*** and designing more effective strategies, considering both objective data (features extracted through FI) and subjective criteria derived from expert judgment.

## Objectives

1. ***Build a hybrid segmentation system:*** Integrate explanatory ***machine learning*** models with a rule-based fuzzy system to segment clients according to their risk profile and likelihood of payment compliance.

2. ***Capture uncertainty and subjectivity:*** Use fuzzy logic to model complex variables such as payment willingness, level of contact, delinquency range, or account age—variables that cannot be fully described using rigid rules.

3. ***Design linguistic segmentation rules:*** Define a set of rules based on expert knowledge and modeled variables (including ML model outputs) to group clients into actionable operational segments.

4. ***Facilitate operational decisions:*** Provide a queryable system that enables collection agents to quickly identify the client type and adapt their management strategies accordingly.

## Project Phases

### Phase 1: Modeling - Machine Learning
- Consolidation of historical and management data
- Training of an explanatory model (XGBoost)
- Generation of class probabilities as input to the fuzzy system

### Phase 2: Modeling - Fuzzy Logic
- Characterization of key variables
- Definition of membership functions
- Creation of inference rules based on expert knowledge
- Conceptual and functional validation of the system

### Phase 3: Integration and Deployment
- Integration of the ML model with the fuzzy system
- Development of a user interface
- Execution of a pilot test with real data

## Table of Contents

```{tableofcontents}
