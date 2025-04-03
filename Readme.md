# Aqua: Responsive Tensor Management for PyTorch
Aqua is a Python module designed to implement responsive offloaded tensors in PyTorch.

## Installation
You can install Aqua using pip:
```bash
pip install -e .
```

## Usage
1. Importing Aqua
```Python
import aqua
```

2. Creating a Responsive Tensor
To convert a PyTorch tensor t into a responsive tensor (rt), use the following line of code:

```Python
rt = aqua.responsive_manager.to_responsive_tensor(t)
```

3. Converting Back to PyTorch Tensor
To retrieve the original PyTorch tensor from a responsive tensor, call the to_torch_tensor() method. Aqua transparently manages where the ```torch_tensor``` is stored. Aqua moves the storage from DRAM to a faster NVLINK device whenever possible.

```Python
torch_tensor = rt.to_torch_tensor()
```

4. Responsiveness in Aqua
At the beginning of your inference loop, make sure to call aqua.responsive_manager.respond():
```Python
aqua.responsive_manager.respond()
```

5. Policies in Aqua: The abstract class ```aqua_policy``` in ```policies/aqua_policy.py``` can be extended to implement custom policies that determine the storage of each ```responsive_tensor```.