import pandas as pd
from src.validate_data import validate_schema

def simulate_corrupted_data(df):
    """Simulates real-world data corruption scenarios across multiple columns."""
    print("[INFO] Injecting simulated faults into data...")
    corrupted_df = df.copy()
    
    # 1. Invalid numerical values (negative tenure)
    corrupted_df.loc[0, 'tenure'] = -5
    
    # 2. Invalid categorical value
    corrupted_df.loc[1, 'Contract'] = 'Lifetime'
    
    # 3. Invalid target label
    corrupted_df.loc[2, 'Churn'] = 'Maybe'
    
    # 4. Datatype mismatch (injecting string into float column)
    corrupted_df.loc[3, 'MonthlyCharges'] = "One Hundred"
    
    # 5. Schema inconsistency (unexpected extra column)
    corrupted_df['unexpected_column'] = 123
    
    # 6. Missing required attribute (dropping a column)
    corrupted_df = corrupted_df.drop(columns=['gender'])
    
    return corrupted_df

if __name__ == "__main__":
    data_path = 'data/raw/churn.csv'
    
    try:
        raw_df = pd.read_csv(data_path)
        
        # Generate corrupted dataframe
        corrupted_df = simulate_corrupted_data(raw_df)
        
        # Pass it to our clean validation module
        print("\n--- Running Validation on Corrupted Data ---")
        validate_schema(corrupted_df, output_report_name="corrupted_simulation_errors.csv")
        
    except FileNotFoundError:
        print(f"[ERROR] Could not find data file at {data_path}.")