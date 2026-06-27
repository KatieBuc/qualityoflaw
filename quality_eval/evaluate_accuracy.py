import os
import json
import pandas as pd
import argparse
from sklearn.metrics import classification_report, confusion_matrix

def load_llm_results(llm_dir_or_file):
    llm_data = []
    
    if os.path.isdir(llm_dir_or_file):
        files = [os.path.join(llm_dir_or_file, f) for f in os.listdir(llm_dir_or_file) if f.endswith('.json')]
    else:
        files = [llm_dir_or_file]

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                filename = data.get("policy_file")
                eval_results = data.get("evaluation_results", {})
                
                for indicator_name, details in eval_results.items():
                    pred_val = 1.0 if details.get("included") == "Yes" else 0.0
                    
                    llm_data.append({
                        "filename": filename,
                        "indicator_id": str(details.get("id")), 
                        "pred_value": pred_val,
                        "indicator_name": indicator_name
                    })
            except Exception as e:
                print(f"Loading JSON failed {file_path}: {e}")
                
    return pd.DataFrame(llm_data)

def calculate_metrics(csv_path, llm_df):
    golden_df = pd.read_csv(csv_path)
    
    golden_df['filename'] = golden_df['filename'].str.strip()
    golden_df['indicator_id'] = golden_df['indicator_id'].astype(str).str.strip()
    golden_df['value'] = golden_df['value'].astype(float)
    
    llm_df['filename'] = llm_df['filename'].str.strip()
    llm_df['indicator_id'] = llm_df['indicator_id'].astype(str).str.strip()
    
    merged_df = pd.merge(
        golden_df, 
        llm_df, 
        on=['filename', 'indicator_id'], 
        how='inner'
    )
    print(golden_df[golden_df['filename'] == "NUSA_TENGGARA_TIMUR_TIMOR_TENGAH_UTARA.txt"])
    
    if merged_df.empty:
        print("Error: No matching data. Please check the CSV and JSON.")
        return None

    print(f"{len(merged_df)} data matched.\n")
    
    y_true = merged_df['value']
    y_pred = merged_df['pred_value']
    
    accuracy = (y_true == y_pred).mean()
    print(f"==================================================")
    print(f"Evaluation Result")
    print(f"==================================================")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Matched Pairs: {len(merged_df)} ")
    print(f"--------------------------------------------------")
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['No (0.0)', 'Yes (1.0)']))
    
    print("Confusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"True Negative: {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"True Positive: {tp}")
    print(f"--------------------------------------------------")
    
    errors = merged_df[merged_df['value'] != merged_df['pred_value']]
    return errors

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Accuracy Checker Against Golden Dataset.")
    parser.add_argument('-g', '--golden_csv', required=True, help="Golden dataset CSV path")
    parser.add_argument('-l', '--llm_input', required=True, help="LLM JSON report, or folder with multiple JSON reports")
    parser.add_argument('-e', '--export_errors', default="error_analysis.csv", help="export file path")
    
    args = parser.parse_args()
    
    print("loading JSON report...")
    llm_df = load_llm_results(args.llm_input)
    if llm_df.empty:
        print("Error: parsing JSON failed")
        return
        
    print("Evaluating...")
    errors_df = calculate_metrics(args.golden_csv, llm_df)
    
    if errors_df is not None and not errors_df.empty:
        error_output = errors_df[['fullname', 'filename', 'indicator_id', 'indicator_value', 'value', 'pred_value']]
        error_output.to_csv(args.export_errors, index=False)
        print(f"Evaluation completed: {args.export_errors}")
    elif errors_df is not None:
        print("The JSON report is completely correct.")

if __name__ == "__main__":
    main()