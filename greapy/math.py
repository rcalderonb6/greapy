import numpy as np

def coth(x):
    """
    Hyperbolic cotangent function.
    
    Parameters
    ----------
    x : float or array_like
        Input value(s)
        
    Returns
    -------
    float or array_like
        Hyperbolic cotangent of x
    """
    return np.cosh(x)/np.sinh(x)

def csch(x):
    """
    Hyperbolic cosecant function.
    
    Parameters
    ----------
    x : float or array_like
        Input value(s)
        
    Returns
    -------
    float or array_like
        Hyperbolic cosecant of x
    """
    return 1 / np.sinh(x)