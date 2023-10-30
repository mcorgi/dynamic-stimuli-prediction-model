from pathlib import Path
import yaml

def load_yaml(file_path: str):
    """
    Load YAML file into a dictionary. 
    Args: file_path (str): 
    Path to the YAML file to be loaded
    """    
    file_path = Path(file_path)
    with file_path.open('r') as f:
        data = yaml.safe_load(f)
    return data

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
def concatenate_dataloaders(dataloaders, verbose=True):
    iterators = {mouse: iter(loader) for mouse,loader in dataloaders.items()}
    total = sum([len(loader) for loader in dataloaders.values()])
    if verbose:
        print('total from concatenate loaders', total)
    count = 0
    while True:
        for mouse, iterator in iterators.items():
            if count > total:
                return
            try:
                yield mouse, next(iterator)
                count += 1
            except StopIteration:
                return