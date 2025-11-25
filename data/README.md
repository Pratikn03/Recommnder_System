# Data directory

- `raw/`: place source CSVs per domain (`fraud/transactions.csv`, `cyber/events.csv`, `behavior/sessions.csv`).
- `interim/`: intermediate cleaned outputs created during preprocessing.
- `processed/`: feature tables ready for modeling.

If no raw files are present, the loaders generate synthetic data so the pipeline and dashboard still run.

Suggested public datasets
- Fraud: IEEE-CIS, Kaggle Credit Card Fraud, PaySim (mobile money)
- Cyber: CIC-IDS, KDDCUP99/NSL-KDD, UNSW-NB15
- Behavior: web/app telemetry, clickstream, or system audit logs
