__version__ = '1.0.0'
__author__ = 'agua-chile'
__description__ = 'Australian Weather Rainfall Prediction Project'

# Import main modules for easy access
from . import data_processing
from . import models
from . import evaluation
from . import visualization

__all__ = [
    'data_processing', 
    'models',
    'evaluation',
    'visualization'
]
