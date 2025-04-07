import pickle
import sys
import os

# Add the path to your project if needed
#sys.path.append('/path/to/adversarial-training')

# Import your function
from reward_function import compute_score

def test_pickle():
    try:
        # Try to pickle the function
        pickled = pickle.dumps(compute_score)
        print("SUCCESS: Function was successfully pickled!")
        
        # Try to unpickle it to make sure it works both ways
        unpickled = pickle.loads(pickled)
        print("SUCCESS: Function was successfully unpickled!")
        
        # Test the unpickled function (optional)
        # result = unpickled("test", "<think>test</think><answer>test</answer>", "test", {"task_id": "test"})
        # print(f"Function execution result: {result}")
        
        return True
    except Exception as e:
        print(f"ERROR: Pickling failed with error: {e}")
        
        # Get more detailed traceback
        import traceback
        print("\nDetailed traceback:")
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    test_pickle()
