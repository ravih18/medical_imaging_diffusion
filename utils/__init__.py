import torch


def print_memory_usage(message):
    print(message)
    print(f"  Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    #print(f"  Reserved Memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

# Function to print parameter differences
def print_param_diff(model, prev_params):
    for name, param in model.named_parameters():
        diff = param.data - prev_params[name]
        print(f"Parameter: {name}, Difference: {diff.norm().item():.6f}")