"""
Package initialization for rainfall prediction project.
"""

__version__ = "1.0.0"
__author__ = "Agua Chile"
__description__ = "Australian Weather Rainfall Prediction Project"

# Import main modules for easy access
from . import config
from . import data_processing
from . import models
from . import evaluation
from . import visualization

__all__ = [
    'config',
    'data_processing', 
    'models',
    'evaluation',
    'visualization'
]