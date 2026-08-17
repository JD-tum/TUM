# Post-Exit Deviations in German Electricity Demand and Residual Load

Master's thesis, Technical University of Munich.
Professor: Prof. Dr. Ziyue Li
Supervisor: Zhenyu Wang
Author: Diehle Jan

Deep-learning analysis of German electricity demand and residual load
around the April 2023 nuclear phase-out. Three parts: mutual information
feature screening, pre- versus post-exit deviation analysis, and a
comparison of four forecasting models.

## Repository contents

| Path | Contents |
|---|---|
| `MasterThesis_New.ipynb` | Full pipeline, all stages, with outputs |
| `smard_data_collector.py` | Downloads the raw SMARD series |
| `data/data.zip` | Raw source files (SMARD, Open-Meteo) |
| `data/thesis_dataset_hourly_v4.csv.zip` | Merged dataset used for results |
| `results/` | Result csv's behind the thesis figures and tables |
| `figures/` | All figures |

## Running the code
The notebook was written for Google Colab with Google Drive mounted.
