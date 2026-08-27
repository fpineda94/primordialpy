import sympy as sp
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Union, Any

# Define a custom type alias for numerical inputs
Numeric = Union[float, np.ndarray]


class Potential(ABC):
    """
    Abstract base class for inflationary potentials.

    This interface defines the fundamental methods required for any
    inflationary potential: the evaluation of the potential itself, 
    and its first and second derivatives with respect to the field.
    """

    @abstractmethod
    def evaluate(self, phi: Numeric) -> Numeric:
        """
        Evaluates the potential V(phi).

        Parameters
        ----------
        phi : float or np.ndarray
            The value(s) of the inflaton field.

        Returns
        -------
        float or np.ndarray
            The potential energy V(phi).
        """
        pass

    @abstractmethod
    def first_derivative(self, phi: Numeric) -> Numeric:
        """
        Evaluates the first derivative of the potential dV/dphi.

        Parameters
        ----------
        phi : float or np.ndarray
            The value(s) of the inflaton field.

        Returns
        -------
        float or np.ndarray
            The first derivative of the potential.
        """
        pass

    @abstractmethod
    def second_derivative(self, phi: Numeric) -> Numeric:
        """
        Evaluates the second derivative of the potential d²V/dphi².

        Parameters
        ----------
        phi : float or np.ndarray
            The value(s) of the inflaton field.

        Returns
        -------
        float or np.ndarray
            The second derivative of the potential.
        """
        pass
    
    def __call__(self, phi: Numeric) -> Numeric:
        """
        Allows calling the object directly as a function: V(phi).

        Parameters
        ----------
        phi : float or np.ndarray
            The value(s) of the inflaton field.

        Returns
        -------
        float or np.ndarray
            The potential energy V(phi).
        """
        return self.evaluate(phi)


class PotentialFunction(Potential):
    """
    Concrete implementation of the Potential interface using symbolic expressions.

    This class relies on SymPy to parse a string representation of the 
    potential, analytically compute its derivatives, and compile them into 
    efficient NumPy-compatible functions.

    Parameters
    ----------
    potential_func : Callable[[Numeric], Numeric]
        Compiled numerical function for V(phi).
    derivative_func : Callable[[Numeric], Numeric]
        Compiled numerical function for dV/dphi.
    second_derivative_func : Callable[[Numeric], Numeric]
        Compiled numerical function for d²V/dphi².
    expr_str : str, optional
        String representation of the symbolic expression. Default is empty.
    param_values : dict, optional
        Dictionary containing the numerical values of the parameters used 
        in the potential. Useful for metadata and reproducibility.

    Examples
    --------
    >>> V = PotentialFunction.function("m**2 * phi**2 / 2", {"m": 1.0})
    >>> V(1.0)
    0.5
    """
    
    def __init__(self, 
                 potential_func: Callable[[Numeric], Numeric], 
                 derivative_func: Callable[[Numeric], Numeric], 
                 second_derivative_func: Callable[[Numeric], Numeric],
                 expr_str: str = "",
                 param_values: Optional[Dict[str, float]] = None):
        
        self._potential_func = potential_func
        self._derivative_func = derivative_func
        self._second_derivative_func = second_derivative_func
        self.expr_str = expr_str 
        self.param_values = param_values or {}

    def __repr__(self) -> str:
        return f"PotentialFunction(expr='{self.expr_str}', params={self.param_values})"
    
    def evaluate(self, phi: Numeric) -> Numeric:
        return self._ensure_array_shape(self._potential_func(phi), phi)

    def first_derivative(self, phi: Numeric) -> Numeric:
        return self._ensure_array_shape(self._derivative_func(phi), phi)
    
    def second_derivative(self, phi: Numeric) -> Numeric:
        return self._ensure_array_shape(self._second_derivative_func(phi), phi)

    @staticmethod
    def _ensure_array_shape(result: Any, phi: Numeric) -> Numeric:
        """
        Helper method to fix a known SymPy lambdify edge case.
        If the derivative is a constant, lambdify might return a scalar 
        even when an array is passed. This ensures shapes match.
        """
        if isinstance(phi, np.ndarray) and np.isscalar(result):
            return np.full_like(phi, result, dtype=float)
        return result

    @classmethod
    def function(cls, potential_expr_str: str, param_values: Optional[Dict[str, float]] = None) -> "PotentialFunction":
        """
        Factory method to instantiate a PotentialFunction from a string.

        Substitutes parameters into the SymPy expression before lambdifying 
        to ensure maximum numerical performance during execution.

        Parameters
        ----------
        potential_expr_str : str
            The mathematical expression of the potential V(phi) as a string.
            The scalar field must be represented by 'phi'.
        param_values : dict, optional
            A dictionary mapping parameter names (str) to their values (float).

        Returns
        -------
        PotentialFunction
            An instance configured with the evaluated numerical functions.

        Raises
        ------
        ValueError
            If required parameters are missing from param_values or if the 
            expression string cannot be parsed.
        """
        if param_values is None:
            param_values = {}
            
        try:
            phi = sp.symbols('phi')
            
            # Parse the string into a sympy expression
            V_expr = sp.sympify(potential_expr_str)
            
            # Extract free symbols, excluding the field 'phi'
            free_symbols = {str(s) for s in V_expr.free_symbols}
            free_symbols.discard('phi') 
            
            # Validate that all required parameters are provided
            provided_params = set(param_values.keys())
            if not free_symbols.issubset(provided_params):
                missing = free_symbols - provided_params
                raise ValueError(f"Missing parameter values for symbols: {missing}")

            # Analytically compute derivatives
            dV_expr = sp.diff(V_expr, phi)
            d2V_expr = sp.diff(dV_expr, phi)

            # Substitute parameter values
            V_expr_sub = V_expr.subs(param_values)
            dV_expr_sub = dV_expr.subs(param_values)
            d2V_expr_sub = d2V_expr.subs(param_values)

            # Lambdify for fast numpy evaluation
            V_func = sp.lambdify(phi, V_expr_sub, modules=['numpy'])
            dV_func = sp.lambdify(phi, dV_expr_sub, modules=['numpy'])
            d2V_func = sp.lambdify(phi, d2V_expr_sub, modules=['numpy'])
            
        except (sp.SympifyError, TypeError) as e:
            raise ValueError(f"Invalid potential expression '{potential_expr_str}': {e}")

        return cls(V_func, dV_func, d2V_func, expr_str=potential_expr_str, param_values=param_values)