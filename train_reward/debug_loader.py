import importlib.util
import sys
import inspect
import pickle
import random
import cloudpickle
from concurrent.futures import ProcessPoolExecutor
import traceback

def worker_function(func, data, solution, reference):
    """Simulate what happens in the process worker"""
    try:
        return func(data, solution, reference)
    except Exception as e:
        return f"Worker error: {str(e)}\n{traceback.format_exc()}"

def process_pool_test():
    """Test with a process pool like the framework"""
    try:
        print("\n=== Testing with ProcessPoolExecutor ===")
        
        # Load the module exactly as the framework does
        file_path = "custom_module.py"
        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function
        compute_score = getattr(module, "compute_score")
        print(f"Function loaded: {compute_score}")
        
        # Try to serialize with cloudpickle (what ProcessPoolExecutor uses)
        try:
            cloudpickled = cloudpickle.dumps(compute_score)
            print(f"Successfully cloudpickled function: {len(cloudpickled)} bytes")
            
            # Try to unpickle
            unpickled = cloudpickle.loads(cloudpickled)
            print("Successfully unpickled function")
        except Exception as e:
            print(f"Cloudpickle failed: {e}")
        
        # Now simulate actual process pool usage
        with ProcessPoolExecutor(max_workers=1) as executor:
            # Test the function in a separate process
            future = executor.submit(
                worker_function, 
                compute_score, 
                "test_data", 
                "<think>Test</think><answer>print('hello')</answer>",
                "test_reference"
            )
            try:
                result = future.result(timeout=10)
                print(f"Process pool result: {result}")
                print("PROCESS POOL TEST SUCCEEDED!")
                return True
            except Exception as e:
                print(f"Process pool execution failed: {e}")
                return False
    except Exception as e:
        print(f"Overall process pool test failed: {e}")
        return False

def framework_simulation():
    """Simulate exactly what the framework does."""
    file_path = "custom_module.py"
    
    print("=== Simulating framework import ===")
    # This is exactly what the framework does
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the function
    compute_score = getattr(module, "compute_score")
    print(f"Function retrieved: {compute_score}")
    
    # Try to pickle it 
    try:
        pickled = pickle.dumps(compute_score)
        print("Successfully pickled function!")
        
        # Try to use it in another process (simulated)
        unpickled = pickle.loads(pickled)
        print("Successfully unpickled function!")
        
        # Test it
        result = unpickled("test", f"<think>Test</think><answer>print('hello')</answer>", "reference")
        print(f"Function result: {result}")
        
        print("FRAMEWORK SIMULATION SUCCEEDED!")
        return True
    except Exception as e:
        print(f"Pickling failed: {e}")
        return False

def test_with_ray():
    """Test with actual Ray serialization."""
    try:
        import ray
        
        if not ray.is_initialized():
            ray.init()
        
        print("\n=== Testing with Ray ===")
        
        # First import the function
        file_path = "custom_module.py"
        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        compute_score = getattr(module, "compute_score")
        
        # Define a Ray remote function that uses compute_score
        @ray.remote
        def remote_scorer(data, solution, reference):
            return compute_score(data, solution, reference)
        
        # Try to use it
        result = ray.get(remote_scorer.remote("test", "<think>Test</think><answer>code</answer>", "ref"))
        print(f"Ray result: {result}")
        print("RAY TEST SUCCEEDED!")
        return True
    except Exception as e:
        print(f"Ray test failed: {e}")
        return False

def debug_import(file_path, function_name):
    """Original debug function."""
    print(f"Attempting to import {function_name} from {file_path}")
    
    try:
        # Print the Python path
        print(f"Python path: {sys.path}")
        
        # Try to import directly as a module name
        print("Attempting direct import...")
        try:
            module = importlib.import_module(file_path.replace('.py', ''))
            print(f"Direct import succeeded: {module}")
        except Exception as e:
            print(f"Direct import failed: {e}")
        
        # Try to import using spec_from_file_location
        print("\nAttempting import from file location...")
        try:
            spec = importlib.util.spec_from_file_location("custom_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"File location import succeeded: {module}")
            
            # Get the function
            func = getattr(module, function_name)
            print(f"Function retrieved: {func}")
            
            # Check if the function can be pickled
            import pickle
            try:
                pickled = pickle.dumps(func)
                print("Function successfully pickled")
                
                # Try to unpickle
                unpickled = pickle.loads(pickled)
                print("Function successfully unpickled")
            except Exception as e:
                print(f"Pickling failed: {e}")
                
                # Get function details
                if hasattr(func, '__module__'):
                    print(f"Function module: {func.__module__}")
                if hasattr(func, '__globals__'):
                    print("Function globals:")
                    for key in func.__globals__:
                        if key.startswith('__'):
                            continue
                        print(f"  {key}: {type(func.__globals__[key])}")
                
        except Exception as e:
            print(f"File location import failed: {e}")
        
        print("\nChecking for 'custom_module' in sys.modules...")
        for name in sys.modules:
            if 'custom' in name.lower():
                print(f"Found module with 'custom' in name: {name}")
        
    except Exception as e:
        print(f"Overall debug failed: {e}")

if __name__ == "__main__":
    # Run all tests
    framework_simulation()
    print("\n" + "="*50 + "\n")
    process_pool_test()  # New test for ProcessPoolExecutor
    print("\n" + "="*50 + "\n")
    test_with_ray()
    print("\n" + "="*50 + "\n")
    debug_import("custom_module.py", "compute_score")
