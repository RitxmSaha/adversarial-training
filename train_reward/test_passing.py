import json
from utils import evaluate_solution

# Load the JSON file
with open('sft_prompts.json', 'r') as f:
    data = json.load(f)

# For each prompt, extract the task_id and solution from the response
for item in data:
    task_id = item['task_id']
    # Extract the solution from between <answer> tags
    response = item['response']
    solution_start = response.find('<answer>') + len('<answer>')
    solution_end = response.find('</answer>')
    solution = response[solution_start:solution_end].strip()
    
    # Evaluate the solution
    print(f"\nEvaluating {task_id}...")
    result = evaluate_solution(task_id, solution)
    if result:
        print(f"Result: {result}")
    else:
        print("Evaluation failed")
