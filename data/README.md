# Data

The pipeline loads the **Titanic dataset** automatically via `seaborn.load_dataset('titanic')`.

If you want to use your own CSV, place it here as `titanic.csv` with these columns:

| Column | Type | Description |
|---|---|---|
| `survived` | int (0/1) | Target variable |
| `sex` | string | male / female |
| `age` | float | Passenger age |
| `sibsp` | int | Siblings/spouses aboard |
| `parch` | int | Parents/children aboard |
| `fare` | float | Ticket fare |
| `embarked` | string | Port of embarkation (S/C/Q) |

Then modify `load_data()` in `pipeline.py` to read from `data/titanic.csv`.
