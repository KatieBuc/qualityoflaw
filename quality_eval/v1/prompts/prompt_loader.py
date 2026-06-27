import argparse
import os
import sys

def generate_judge_prompt(criteria_file_path, template_file_path, policy_file_path):
    for file_path in [criteria_file_path, template_file_path]:
        if not os.path.exists(file_path):
            print(f"Cannot find the file: {file_path}", file=sys.stderr)
            sys.exit(1)

    criteria_list_str = ""
    with open(criteria_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split("|")
            if len(parts) == 3:
                cid = parts[0].strip()
                indicator = parts[1].strip()
                question = parts[2].strip()
                
                criteria_list_str += f"- {cid} ({indicator}): {question}\n"
            else:
                print(f"Error in line {line_num}, continue to '{line}'", file=sys.stderr)

    with open(policy_file_path, 'r', encoding='utf-8') as f:
        policy = f.read()
    
    with open(template_file_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()


    final_prompt = prompt_template.replace("{{CRITERIA_LIST}}", criteria_list_str).replace("{{POLICY_TEXT}}", policy)
    
    return final_prompt

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    criteria_file_path = os.path.join(current_dir, "01_scope_of_violence.txt")
    template_path = os.path.join(current_dir, "prompt_template.txt")
    policy_file_path = os.path.join(current_dir, "../translated_policy\ACEH_BIREUEN.txt")

    final_prompt = generate_judge_prompt(
        criteria_file_path=criteria_file_path,
        template_file_path=template_path,
        policy_file_path=policy_file_path
    )

    print(final_prompt[:8000])

if __name__ == "__main__":
    main()