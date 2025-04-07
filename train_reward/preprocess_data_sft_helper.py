import os
import pandas as pd
from evalplus.data import get_human_eval_plus
import argparse

def format_prompt_with_test_cases(original_prompt, proxy_test_cases):
    """
    Replaces examples in the original prompt with test cases.
    """
    # Find where the examples start in the original prompt
    lines = original_prompt.strip().split('\n')
    example_index = -1
    indent = "    "  # Default indent
    
    # Extract function name from the signature
    function_name = None
    for line in lines:
        if line.strip().startswith('def '):
            # Extract function name from "def function_name(params):"
            function_signature = line.strip()
            function_name = function_signature.split('def ')[1].split('(')[0].strip()
            break
    
    # If function name not found, default to original behavior
    if not function_name:
        return original_prompt
    
    # Find the first example line and its indentation
    for i, line in enumerate(lines):
        if '>>>' in line:
            example_index = i
            indent = line[:line.find('>>>')]
            break
    
    # If no examples found, return original prompt
    if example_index == -1:
        return original_prompt
    
    # Keep everything before the examples
    result_lines = lines[:example_index]
    
    # Add evaluation statement before examples
    evaluation_statement = f"\n{indent}The following examples will be used for evaluation:"
    result_lines.append(evaluation_statement)
    
    # Extract test cases from proxy_test_cases
    test_cases = []
    in_check_function = False
    
    for line in proxy_test_cases.split('\n'):
        if line.strip().startswith('def check(candidate):'):
            in_check_function = True
        elif in_check_function and line.strip().startswith('assert candidate('):
            test_cases.append(line.strip())
    
    # Format test cases as examples with proper indentation
    for tc in test_cases:
        tc = tc.replace('assert ', '')
        if ' == ' in tc:
            # Replace "candidate" with the actual function name
            call, expected = tc.split(' == ')
            call = call.replace('candidate', function_name)
            result_lines.append(f"{indent}>>> {call}")
            result_lines.append(f"{indent}{expected}")
    
    # Find where docstring ends to add remaining content
    docstring_end = -1
    for i in range(example_index, len(lines)):
        if '"""' in lines[i] or "'''" in lines[i]:
            docstring_end = i
            break
    
    # Add closing docstring and any content after it
    if docstring_end != -1:
        result_lines.append(lines[docstring_end])
        result_lines.extend(lines[docstring_end+1:])
    else:
        result_lines.append(f'{indent}"""')
    
    return '\n'.join(result_lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./sft_data')
    parser.add_argument('--num_examples', type=int, default=10)
    parser.add_argument('--convert', action='store_true', help='Convert filled CSV to JSON')
    
    args = parser.parse_args()
    
    # Get HumanEval data
    human_eval_plus = get_human_eval_plus()
    
    # Create a list of examples
    dataset_list = []
    for task_id, problem in human_eval_plus.items():
        original_prompt = problem.get('prompt', '')
        proxy_test_cases = problem.get('test', '')
        canonical_solution = problem.get('canonical_solution', '').strip()
        
        # Format the prompt with test cases as examples
        reformatted_function = format_prompt_with_test_cases(original_prompt, proxy_test_cases)
        
        # Format the prompt in the conversation style you requested
        formatted_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
            "The assistant first thinks about the reasoning process in the mind and then provides the user\n"
            "with the answer. The reasoning process and answer are enclosed within <think> </think> and\n"
            "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>\n"
            "<answer> answer here </answer>.\n\n"
            
            "Implement the following function:\n\n"
            f"{reformatted_function}\n\n"
        )
        
        # Format the response with empty think tags and the canonical solution in answer tags
        formatted_response = f"<think></think>\n<answer>\n{canonical_solution}\n</answer>"
        
        dataset_list.append({
            'task_id': task_id,
            'prompt': formatted_prompt,
            'response': formatted_response
        })
        
        # Only take the first n examples
        if len(dataset_list) >= args.num_examples:
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset_list)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(args.output_dir, 'sft_prompts.json')
    df.to_json(json_path, orient='records', indent=2)
    
    # Save as JSON files for training and testing
    train_path = os.path.join(args.output_dir, 'train.json')
    test_path = os.path.join(args.output_dir, 'test.json')
    
    df.to_json(train_path, orient='records', indent=2)
    df.to_json(test_path, orient='records', indent=2)  # Using same data for test for simplicity
    
    print(f"Created JSON file with {args.num_examples} examples at: {json_path}")
    print(f"Created JSON files for training at: {train_path}")
    print(f"Created JSON files for testing at: {test_path}")
    print("The responses contain empty <think> tags and the canonical solutions in <answer> tags.")
    
    # Convert functionality for JSON instead of parquet
    def convert_csv_to_json(csv_path, train_path, test_path=None):
        """Convert CSV to JSON files for training/testing."""
        df = pd.read_csv(csv_path)
        
        # Save training data
        df.to_json(train_path, orient='records', indent=2)
        print(f"Saved training data to {train_path}")
        
        # Save test data (if path provided)
        if test_path:
            df.to_json(test_path, orient='records', indent=2)
            print(f"Saved test data to {test_path}")
    
    if args.convert:
        train_path = os.path.join(args.output_dir, 'train.json')
        test_path = os.path.join(args.output_dir, 'test.json')
        convert_csv_to_json(json_path, train_path, test_path)
