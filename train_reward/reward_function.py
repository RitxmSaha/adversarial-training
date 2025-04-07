import re
import wandb

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Absolutely minimal reward function with no dependencies.
    """
    score = 0.5
    print(f"Computing minimal score for solution of length {len(solution_str)}")
    
    try:
        wandb.log({'reward/fixed_score': score})
    except Exception as e:
        print(f"Could not log to wandb: {e}")
    
    return score

# Do not include any other functions or unused imports