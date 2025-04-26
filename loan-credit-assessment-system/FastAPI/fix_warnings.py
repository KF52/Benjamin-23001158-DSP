import numpy as np
import warnings

# Monkey patch np.bool to maintain compatibility with SHAP
if not hasattr(np, 'bool'):
    print("== XAI DEBUG MSG == Adding np.bool compatibility for SHAP")
    np.bool = bool

    # Suppress the deprecation warning
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

# Suppress IPython warning
warnings.filterwarnings('ignore', message='IPython could not be loaded')