import os
import json
import sys
import argparse
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from openai import OpenAI
from quality_eval.v1.prompts.prompt_loader import generate_judge_prompt
from dotenv import load_dotenv
import traceback
from datetime import datetime

CRITERIA_FILES = [
    "01_scope_of_violence.txt",
    "02_institutional_mechanism.txt",
    "03_specialised_support_services.txt",
    "04_primary_prevention.txt",
    "05_stakeholder_engagement.txt",
    "06_intersectional_approach.txt",
    "07_budget_funding_sources.txt"
]

class CriterionResult(BaseModel):
    id: str = Field(description="The ID of the indicator being evaluated. Only contain numbers and a single dot for seperation.")
    indicator: str = Field(description="The name of the indicator being evaluated.")
    included: str = Field(description="Must be 'Yes' or 'No'.")
    evidence: Optional[str] = Field(default=None, description="Exact quote from the text if included is 'Yes', otherwise null.")
    rationale: str = Field(description="Explanation of how the text addresses or fails to address this indicator.")

class PolicyEvaluationResponse(BaseModel):
    evaluation_results: List[CriterionResult]

load_dotenv()

def run_policy_evaluation(prompt: str) -> dict:
    client = OpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),  
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    
    deployment_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o") 
    
    print("requesting...")
    # print(prompt)
    
    response = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format=PolicyEvaluationResponse, 
        temperature=0.1
    )

    parsed_output = response.choices[0].message.parsed
    if parsed_output:
        output_dict = parsed_output.model_dump()
        
        # Reconstruct the original Dict structure if needed by your application
        output_dict['evaluation_results'] = {
            item['id']: item for item in output_dict['evaluation_results']
        }
        
        # print(output_dict)
        return output_dict
    else:
        raise ValueError("LLM return structure error.")

def main():
    parser = argparse.ArgumentParser(description="policy quality evaluation.")
    parser.add_argument('-p', '--policy', required=True, help="filepath of policy document.")
    parser.add_argument('-o', '--output_folder', required=True, help="output folder of the evalutaion.")
    parser.add_argument('-c', '--criteria_folder', required=True, help="base folder of the criteria text file.")
    parser.add_argument('-t', '--template',  required=True, help="path of the prompt template text file.")

    args = parser.parse_args()

    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Error: API key not found.")
        return

    if not os.path.exists(args.policy):
        print(f"Error: Cannot find the file: {args.policy}")
        return

    final_report = {
        "policy_file": os.path.basename(args.policy),
        "evaluation_results": {}
    }

    print(f"starting for {len(CRITERIA_FILES)} categories of criterias...\n")

    for idx, criteria_file in enumerate(CRITERIA_FILES, 1):
        print(f"--------------------------------------------------")
        print(f"[{idx}/{len(CRITERIA_FILES)}] Processing: {criteria_file}")
        
        try:
            final_prompt = generate_judge_prompt(
                criteria_file_path=os.path.join(args.criteria_folder, criteria_file), 
                template_file_path=args.template,
                policy_file_path=args.policy
            )
            
            batch_result = run_policy_evaluation(final_prompt)
            batch_evals = batch_result.get("evaluation_results", {})

            for k, item in batch_evals.items():
                if k in final_report["evaluation_results"]:
                    print(f"Warning: ID '{k}' ({item['indicator']}) is already exist, it will be renew by new outcome.")

            final_report["evaluation_results"].update(batch_evals)
            
            print(f"{criteria_file} completed")
            
        except Exception as e:
            print(f"error when processing {criteria_file} : {e}", file=sys.stderr)
            traceback.print_exc()

    print(f"\n==================================================")
    output_name = now = datetime.now()
    dt = now.strftime("%d%m%Y%H%M%S")
    basename = os.path.basename(args.policy).replace(".txt", ".json")
    output_name = dt + '-' + os.getenv("AZURE_OPENAI_MODEL") + '-' + basename
    output_path = os.path.join(args.output_folder, output_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
        
    print(f"All evaluation completed, output: {output_path}")

if __name__ == "__main__":
    main()