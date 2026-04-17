# Data-Driven Surrogate Modelling and Multi-Objective Optimization of WEDM Parameters for Ti-6Al-4V Alloy

**BTP-II Report | B.Tech + M.Tech Dual Degree Project**

**Author:** Sagar Chandan (22ME31051)
**Supervisor:** Dr. Sankha Deb
**Department:** Mechanical Engineering, Indian Institute of Technology Kharagpur
**Date:** April 2026

---

## Overview

This repository contains the complete code, data, and report for the **Bachelor's Thesis Project Phase II (BTP-II)**, which extends the hybrid RSM-ML modelling framework developed in [BTP-I](https://github.com/sagarchandan/BTP-I) (Ti-6Al-7Nb) to **Ti-6Al-4V alloy** with multi-objective optimization and experimental validation.

### What this project does

1. **RSM-based data expansion:** Takes a 9-point Taguchi L9 experimental dataset and expands it to ~1,100 points using Response Surface Methodology equations
2. **ML surrogate training:** Trains Random Forest, SVR, and XGBoost models as surrogates for Material Removal Rate (MRR) and Surface Roughness (SR)
3. **Multi-objective optimization:** Uses NSGA-II with each surrogate to produce Pareto-optimal trade-off solutions
4. **Knee-point analysis:** Identifies the balanced compromise parameter set from the Pareto front
5. **Experimental validation:** Validates the predicted optimum on a physical WEDM machine at IIT Kharagpur

### Key Results

| Metric | SVR (Best) | RF | XGBoost |
|--------|-----------|-----|---------|
| CV R² (MRR) | 0.9997 | 0.9737 | 0.9917 |
| CV R² (SR) | 0.9997 | 0.9924 | 0.9942 |
| Hypervolume | 1.014 | 0.987 | 1.004 |

**Recommended Parameters (Knee-point):** Ip = 20 A, Ton = 110 µs, Toff = 60 µs, Vs = 233 V

**Experimental Validation:**
- MRR: 7.50 mm³/min (predicted 7.98, error 6.0%)
- SR: 1.74 µm (predicted 1.97, error 11.7%)

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── report/
│   ├── main.tex              # Complete LaTeX source
│   ├── BTP2_Report.pdf       # Compiled report (29 pages)
│   └── iitkgp_logo.png       # IIT KGP logo for title page
├── code/
│   └── generate_plots.py     # Plot generation script
├── data/
│   ├── expanded_dataset_1100.csv    # RSM-expanded training data
│   ├── cv_results.csv               # 5-fold CV metrics
│   ├── validation_results.csv       # L9 validation metrics
│   ├── hypervolume_comparison.csv   # Pareto front quality
│   ├── pareto_RF.csv                # RF Pareto solutions
│   ├── pareto_SVR.csv               # SVR Pareto solutions
│   ├── pareto_XGBoost.csv           # XGBoost Pareto solutions
│   └── all_pareto_solutions.csv     # Combined Pareto data
├── plots/
│   ├── 01_methodology.png
│   ├── 02_experimental_data.png
│   ├── 03_parameter_effects.png
│   ├── 04_expanded_dataset.png
│   ├── 05_cv_comparison.png
│   ├── 06_predicted_vs_actual.png
│   ├── 07_validation_table.png
│   ├── 08_pareto_with_kneepoints.png
│   ├── 09_hypervolume.png
│   ├── 10_kneepoint_table.png
│   ├── 11_mrr_sr_correlation.png
│   └── 12_experimental_validation.png
└── experimental/
    ├── wedm_machine.jpg       # Electronica Job Master D-zire
    ├── wedm_cutting.jpg       # WEDM cutting with flushing
    ├── wedm_controller.jpg    # CNC controller panel
    ├── wedm_wireguide.jpg     # Wire guide close-up
    └── sr_measurement.png     # Taylor Hobson Talysurf measurement
```

## Methodology

```
L9 Data (9 pts) → RSM Equations → Expanded Dataset (1,100 pts) → ML Surrogates → NSGA-II → Knee-point → Experimental Validation
```

![Methodology](plots/01_methodology.png)

## Dataset

The primary experimental data is sourced from:

> Jagdale, S. N. et al. (2025). "Experimental investigation of process parameters in Wire-EDM of Ti-6Al-4V." *Scientific Reports*, 15, 5652.

| Parameter | Low | Mid | High | Unit |
|-----------|-----|-----|------|------|
| Peak Current (Ip) | 20 | 25 | 30 | A |
| Pulse-on Time (Ton) | 110 | 115 | 120 | µs |
| Pulse-off Time (Toff) | 50 | 55 | 60 | µs |
| Servo Voltage (Vs) | 220 | 230 | 240 | V |

## Experimental Setup

Validation experiments were conducted on an **Electronica Job Master D-zire** CNC Wire EDM machine (Electronica HiTech Machine Tools Pvt. Ltd., SRP Electronica Group) at the Training Workshop, Department of Mechanical Engineering, IIT Kharagpur. Surface roughness was measured using a **Taylor Hobson Talysurf** profilometer with TalyMap Gold 7.1 software.

## Dependencies

```
numpy
pandas
scikit-learn
xgboost
matplotlib
pymoo
joblib
```

Install with:
```bash
pip install -r requirements.txt
```

## Compiling the Report

```bash
cd report
pdflatex main.tex
pdflatex main.tex  # Run twice for TOC and references
```

## Citation

If you use this work, please cite:

```bibtex
@thesis{chandan2026btp2,
  title   = {Data-Driven Surrogate Modelling and Multi-Objective Optimization of WEDM Parameters for Ti-6Al-4V Alloy},
  author  = {Chandan, Sagar},
  year    = {2026},
  school  = {Indian Institute of Technology Kharagpur},
  type    = {B.Tech + M.Tech Dual Degree Project (BTP-II)},
  note    = {Supervisor: Dr. Sankha Deb}
}
```

## Related Work

- **BTP-I:** Data-Driven Modeling and Optimization of WEDM Parameters for Ti-6Al-7Nb (Nov 2025)

## Acknowledgements

- **Supervisor:** Dr. Sankha Deb, Department of Mechanical Engineering, IIT Kharagpur
- **Technical Assistance:** Mr. Gobinda Chandra Behera, Research Scholar

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
