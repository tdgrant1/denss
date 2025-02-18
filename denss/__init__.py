# denss/__init__.py

try:
    from ._version import __version__
except ImportError:
    __version__ = ''

from .core import *  # Import EVERYTHING from core.py
from .options import * # Import EVERYTHING from options.py

__all__ = [] # Initialize __all__ as an empty list

# Dynamically populate __all__ with names from core.py and options.py
import inspect
from . import core, options # Import core and options modules to inspect
modules_to_inspect = [core, options] # List of modules to inspect

for module in modules_to_inspect:
 for name, obj in inspect.getmembers(module):
     if not name.startswith('_'): # Exclude private names (starting with _)
         if inspect.isclass(obj) or inspect.isfunction(obj): # Include classes and functions
             __all__.append(name)
