# Table of contents

- [Contributing to LightAutoML](#contriburing-to-lightautoml)
- [Writing Documentation](#writing-documentation)
- [Style Guide](#style-guide) 

## Contributing to LightAutoML

Thank you for your interest in contribution to LightAutoML! Before you begin writing code, it is
important that you share your intention to contribute with developers team.

- First please look for discussions on this topic in [issues](https://github.com/sberbank-ai-lab/LightAutoML/issues)
before implementing anything inside the project.
- Pick an issue and comment that you would like to work on it.
- If there is no discussion on this topic, create one. 
  Please, include as much information as you can,
  any accompanying data (your tests, may be articles),
  and may be your proposed solution. 
- If you need more details, please ask we will provide it ASAP.

Once you implement and test your feature or bug-fix, please submit
a Pull Request to https://github.com/sberbank-ai-lab/LightAutoML.

## Codebase structure

- [docs](docs) - For documenting we use [Sphinx](https://www.sphinx-doc.org/).
  It provides easy to use auto-documenting via docstrings.
  - [Tutorials](docs/notebooks) - Notebooks with tutorials.
    
- [lightautoml](lightautoml) - The code of LightAutoML library.
    - [addons](lightautoml/addons) - Extensions of core functionality.
    - [automl](lightautoml/automl) - The main module, which includes the AutoML class,
      blenders and ready-made presets.
    - [dataset](lightautoml/dataset) - The internal interface for working with data.
    - [image](lightautoml/image) - The internal interface for working with image data.
    - [ml_algo](lightautoml/ml_algo) - Modules with machine learning algorithms
      and hyperparameters tuning tools.
    - [pipelines](lightautoml/pipelines) - Pipelines for different tasks (feature processing & selection).
    - [reader](lightautoml/reader) - Utils for training and analysing data.
    - [report](lightautoml/report) - Report generators and templates.
    - [tasks](lightautoml/tasks) - Define the task to solve its loss, metric.
    - [text](lightautoml/text) - The internal interface for working with text data.
    - [transformers](lightautoml/transformers) - Feature transformations.
    - [utils](lightautoml/utils) - Common util tools (Timer, Profiler, Logging).
    - [validation](lightautoml/validation) - Validation module.
    

## Developing LightAutoML

### Installation

If you are installing from the source, you will need Python 3.6.12 or later.
We recommend you to install an [Anaconda](https://www.anaconda.com/products/individual#download-section)
to work with environments. 


1. Once you install Anaconda, you need to set your own environment:
```bash
conda create -n Pyth36 python=3.6
conda activate Pyth36
```

2. To clone the project to your own local machine:
```bash
git clone https://github.com/sberbank-ai-lab/LightAutoML
cd LightAutoML
```

3. Install LightAutoML in develop mode:
```bash
./build_package.sh
source ./lama_venv/bin/activate
poetry install
```

After that there is ```lama_venv``` environment, where you can test and implement your own code.
So, you don't need to rebuild all project every time.

### Testing

Before PR, please check your code by:

```bash
bash test_package.sh
```
It takes all ```demo*.py``` and use ```pytest``` to run it. Please, add your own tests.

## Style Guide

We use [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## Documentation



### Building Documentation:

To build the documentation:

1. Clone repository to your device.

2. Make environment and install requirements

```bash
python3 -m venv docs_venv
source docs_venv/bin/activate
cd docs
pip install -r requirements.txt
pip install sphinx-rtd-theme
```
3. Generate HTML documentation files. The generated files will be in `docs/_build/html`
```bash
cd docs
make clean html
```


### Writing Documentation

There are some rules, that docstrings should fit.

1. LightAutoML uses [Google-style docstring formatting](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). Length of line inside docstring must be limited to
   80 characters to fit into Jupyter documentation popups.
   
2. Every non one-line docstring should have a paragraph at its end, regardless of where it will be used:
   in the documentation for a class, module, function, class method etc.
   One-liners shouldn't have a paragraph at its end.
   Also, if you don't have special fields like Note, Warning you may don't add a paragraph at its end.
   
3. Once you added some module to LightAutoML, you should add some info about it at the beginning of the module.
   Example of this you can find in `docs/mock_docs.py`.
   Also, if you use submodules, please add description to `__init__.py`

4. There is an example for documenting standalone functions.

```python3

### imports

from typing import List, Union

import numpy as np
import torch

### <...>

def typical_function(a: int, b: Union['np.ndarray', None] = None) -> List[int]:
    """Short function description, terminated by dot.

    Some details. The parameter after arrow is return value type.

    Use 2 newlines to make a new paragraph,
    like in `LaTeX by Knuth <https://en.wikipedia.org/wiki/LaTeX>`_.

    Args:
        a: Parameter description, starting with a capital
          latter and terminated by a period.
        b: Textual parameter description.

    .. note::
        Some additional notes, with special block.

        If you want to itemize something (it is inside note):

            - First option.
            - Second option.
              Just link to function :func:`torch.cuda.current_device`.
            - Third option.
              Also third option.
            - It will be good if you don`t use it in args.

    .. warning::
        Some warning. Every block should be separated
        with other block with paragraph.

    .. warning::
        One more warning. Also notes and warnings
        can be upper in the long description of function
        
    Example:
        
        >>> print('MEME'.lower())
        meme
        >>> b = typical_function(1, np.ndarray([1, 2, 3]))

    Returns:
        Info about return value.

    Raises:
        Exception: Exception description.
        
    """

    return [a, 2, 3]
```

5. Docstring for generator function.
```python3

def generator_func(n: int):
    """Generator have a ``Yields`` section instead of ``Returns``.
    
    Args:
        n: Number of interations.
        
    Yields:
        The next number in the range of ``0`` to ``n-1``.
    
    Example:
        Example description.
    
        >>> print([i for i in generator_func(4)])
        [0, 1, 2, 3]
    
    """

```
6. Documenting classes.
```python3

### imports

from typing import List, Union

import numpy as np
import torch

### <...>

class ExampleClass:
    """The summary line for a class that fits only one line.
    
    Long description. 
    
    If the class has public attributes, they may be documented here
    in an ``Attributes`` section, like in ``Args`` section of function.

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method. Use arrow to set the return type.
    
    On the stage before __init__ we don't know anything about `Attributes`,
    so please, add description about it's types.
    
    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.
    
    """

    def __init__(self, param1: int, param2: 'np.ndarray', *args, **kwargs):
        """Example of docstring of the __init__ method.
        
        Note:
            You can also add notes as ``Note`` section.
            Do not include the `self` parameter in the ``Args`` section.
            
        Args:
            param1: Description of `param1`.
            param2: Description of `param2`.
            *args: Description of positional arguments.
            **kwargs: Description of key-word arguments.
        
        """
        self.attr1 = param1
        self.attr2 = param2
        if len(args) > 0:
            self.attr2 = args[0]
        self.attr3 = kwargs # will not be documented.
        self.figure = 4 * self.attr1
        
    @property
    def readonly_property(self) -> str:
        """Properties should be documented in 
        their getter method.
        
        """
        return 'lol'
    
    @property
    def readwrite_property(self) -> List[str]:
        """Properties with both a getter and setter
        should only be documented in their getter method.
        
        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return [str(self.figure)]

    @readwrite_property.setter
    def readwrite_property(self, value: int):
        self.figure = value

    def some_method(self, param1: int, param2: float = np.pi) -> List[int]:
        """Just like a functions.
        
        Long description.
        
        .. warning:
            This method do somethinpip install sphinx-rtd-themeg. May be undefined-behaviour.
        
        Args:
            param1: Some description of param1.
            param2: Some description of param2. Default value
               will be contained in signature of function.
        
        Returns:
            Array with `1`, `2`, `3`.
            
        """
        self.attr1 = param1
        self.attr2 += param2
        
        return [1, 2, 3]
    
    
    def __special__(self):
        """By default we aren`t include dundered members.
        
        Also there may be no docstring.
        """
        
    def _private(self):
        """By default we aren`t include private members.
        
        Also there may be no docstring.
        """
        
    @staticmethod
    def static_method(param1: int):
        """Description of static method.
        
        Note:
            As like common method of class don`t use `self`.
            
        Args:
            param1: Description of `param1`.
            
        """
        print(param1)

```

[comment]: <> (7. Some tips about typing.)

[comment]: <> (```python3)

[comment]: <> (### Please don't use TYPE_CHECKING option)

[comment]: <> (# from typing import TYPE_CHECKING)

[comment]: <> (from typing import Union, List, Optional, TypeVar)




[comment]: <> (```)



### Adding new submodules

For the description of the module to be included in the documentation,
you should set the variable ```autosummary_generate = True``` in ```docs/conf.py```
before generating the documentation. This will generate a draft
of the documentation for your code.

### Adding Tutorials

We use ```nbsphinx``` Sphinx extension for tutorials. Examples, you can find in ```docs/notebooks```.
Please put your tutorial in this folder and after add it in ```docs/Tutorials.rst```.

