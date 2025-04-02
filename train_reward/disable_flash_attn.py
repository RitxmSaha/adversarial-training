import os
import sys
import types
import importlib.util

# Force environment variables
os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

# Define a function to patch the _ulysses_flash_attention_forward in verl
def patch_monkey():
    # Find the module path
    verl_path = None
    for path in sys.path:
        if os.path.exists(os.path.join(path, "verl")):
            verl_path = os.path.join(path, "verl")
            break
    
    if verl_path is None:
        print("Could not find verl module path")
        return
    
    # Define the target path
    target_path = os.path.join(verl_path, "models", "transformers", "monkey_patch.py")
    
    if not os.path.exists(target_path):
        print(f"Could not find {target_path}")
        return
    
    # Load the module
    spec = importlib.util.spec_from_file_location("monkey_patch", target_path)
    monkey_patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monkey_patch)
    
    # Define a replacement function that skips flash attention
    def no_flash_attention_forward(*args, **kwargs):
        print("Flash attention disabled, using standard attention")
        # Use the standard attention implementation
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        # Call the standard attention method
        try:
            # First, try using the normal attention method
            return Qwen2Attention._attention_forward(*args, **kwargs)
        except AttributeError:
            # If that fails, try the __call__ method (will be slow but should work)
            q, k, v = args[0], args[1], args[2]
            attention = Qwen2Attention(monkey_patch.config)
            return attention(q, k, v, **kwargs)
    
    # Replace the function
    monkey_patch._ulysses_flash_attention_forward = no_flash_attention_forward
    
    print("Successfully patched _ulysses_flash_attention_forward to disable flash attention")

# Run the patch
patch_monkey()

# Now proceed with normal execution
