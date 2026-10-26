import os
import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column

def validate_total_charges(series):
    values = series.astype("string").str.strip()
    return values.isna() | values.eq("") | pd.to_numeric(values, errors="coerce").notna()

def get_telco_schema():
    return pa.DataFrameSchema(
        {
            "customerID": Column(pa.String),
            "gender": Column(pa.String, Check.isin(["Male", "Female"])),
            "SeniorCitizen": Column(pa.Int, Check.isin([0, 1])),
            "Partner": Column(pa.String, Check.isin(["Yes", "No"])),
            "Dependents": Column(pa.String, Check.isin(["Yes", "No"])),
            "tenure": Column(pa.Int, Check.ge(0)),
            "PhoneService": Column(pa.String, Check.isin(["Yes", "No"])),
            "MultipleLines": Column(pa.String, Check.isin(["Yes", "No", "No phone service"])),
            "InternetService": Column(pa.String, Check.isin(["DSL", "Fiber optic", "No"])),
            "OnlineSecurity": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
            "OnlineBackup": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
            "DeviceProtection": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
            "TechSupport": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
            "StreamingTV": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
            "StreamingMovies": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
            "Contract": Column(pa.String, Check.isin(["Month-to-month", "One year", "Two year"])),
            "PaperlessBilling": Column(pa.String, Check.isin(["Yes", "No"])),
            "PaymentMethod": Column(pa.String, Check.isin(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])),
            "MonthlyCharges": Column(pa.Float, Check.ge(0.0)),
            # "TotalCharges": Column(pa.String, nullable=True),
            # "TotalCharges": Column(None, Check(lambda s: pd.to_numeric(s, errors="coerce").notna() | s.isna()), nullable=True),
            "TotalCharges": Column(None, Check(validate_total_charges), nullable=True),
            "Churn": Column(pa.String, Check.isin(["Yes", "No"])),
        },
        strict=True,
    )


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_telco_schema()

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        failures = err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]
        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False