import sympy as sp
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Union




class Potential(ABC):
    """
    Abstract base class for inflationary potentials.
    """

    @abstractmethod
    def evaluate(self, phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the value of the potential V(phi)."""
        pass

    @abstractmethod
    def first_derivative(self, phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns dV/dphi."""
        pass

    @abstractmethod
    def second_derivative(self, phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns d²V/dphi²."""
        pass
    
    def __call__(self, phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Allows calling the object directly as a function: V(phi)."""
        return self.evaluate(phi)


class PotentialFunction(Potential):
    """
    Concrete implementation of the Potential interface using symbolic expressions.
    """

    def __init__(self, 
                 potential_func: Callable, 
                 derivative_func: Callable, 
                 second_derivative_func: Callable,
                 expr_str: str = ""):
        
        self._potential_func = potential_func
        self._derivative_func = derivative_func
        self._second_derivative_func = second_derivative_func
        self.expr_str = expr_str 

    def evaluate(self, phi):
        return self._potential_func(phi)

    def first_derivative(self, phi):
        return self._derivative_func(phi)
    
    def second_derivative(self, phi):
        return self._second_derivative_func(phi)

    @classmethod
    def function(cls, potential_expr_str: str, param_values: Optional[Dict[str, float]] = None):
        """
        Factory method optimized: substitutes parameters BEFORE lambdifying.
        """
        if param_values is None:
            param_values = {}
            
        try:
            phi = sp.symbols('phi')
            
            V_expr = sp.sympify(potential_expr_str)
            
            free_symbols = {str(s) for s in V_expr.free_symbols}
            free_symbols.discard('phi') 
            
            provided_params = set(param_values.keys())
            if not free_symbols.issubset(provided_params):
                missing = free_symbols - provided_params
                raise ValueError(f"Faltan valores para los parámetros: {missing}")

        
            dV_expr = sp.diff(V_expr, phi)
            d2V_expr = sp.diff(dV_expr, phi)

 
            V_expr_sub = V_expr.subs(param_values)
            dV_expr_sub = dV_expr.subs(param_values)
            d2V_expr_sub = d2V_expr.subs(param_values)

    
            V_func = sp.lambdify(phi, V_expr_sub, modules=['numpy'])
            dV_func = sp.lambdify(phi, dV_expr_sub, modules=['numpy'])
            d2V_func = sp.lambdify(phi, d2V_expr_sub, modules=['numpy'])
            
        except Exception as e:
            raise ValueError(f"Error interpretando el potencial: {e}")

        return cls(V_func, dV_func, d2V_func, expr_str=potential_expr_str)
