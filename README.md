# WEDM Surrogate Modelling & Multi-Objective Optimization

## Overview
This repository contains the code, data, and reports for a two-phase BTP 
investigating data-driven surrogate modelling and multi-objective optimization 
of Wire EDM (WEDM) machining parameters for titanium alloys.

## BTP-I — Ti-6Al-7Nb
- **Surrogates**: Random Forest, SVR, XGBoost, ANN
- **Optimization**: NSGA-II, PSO
- **Decision Analysis**: TOPSIS, Knee-point detection, SHAP interpretability
- **Dataset**: 54-run Taguchi DOE → 729-point RSM-augmented synthetic DOE

## BTP-II — Ti-6Al-4V
- **Surrogates**: Random Forest, SVR, XGBoost
- **Optimization**: NSGA-II
- **Decision Analysis**: Knee-point detection (Euclidean distance)
- **Dataset**: Jagdale et al. (2025) L9 Taguchi → RSM-augmented DOE

## Key Methodology
RSM-augmented data generation strategy to overcome the scarcity of 
experimental data for expensive-to-machine biomedical titanium alloys, 
enabling reliable ML surrogate training from small-scale DOEs.

## Requirements
- Python 3.x, scikit-learn, XGBoost, pymoo, numpy, pandas, matplotlib
