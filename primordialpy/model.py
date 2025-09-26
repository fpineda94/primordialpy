import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

# Se usan unidades naturales M_p = 1
# Definición de la clase base para potenciales

class Potential(ABC):

    """
    Abstract base class for inflationary potentials.

    This class defines the interface that any specific inflationary potential must implement.
    It ensures consistency when evaluating the potential and its derivatives, which are 
    required for solving background and perturbation equations in inflationary models.

    Methods
    -------
    evaluate(phi)
        Returns the value of the potential V(phi) at the field value phi.

    first_derivative(phi)
        Returns the first derivative of the potential with respect to phi, dV/dphi.

    second_derivative(phi)
        Returns the second derivative of the potential with respect to phi, d²V/dphi².
    """

    @abstractmethod
    def evaluate(self, phi):
        pass

    @abstractmethod
    def first_derivative(self, phi):
        pass

    @abstractmethod
    def second_derivative(self, phi):
        pass




class PotentialFunction(Potential):

    """
    Concrete implementation of the Potential interface using symbolic expressions.

    This class allows defining inflationary potentials from arbitrary symbolic expressions
    (as strings) and converts them into callable numerical functions using SymPy and NumPy.
    It also supports plotting the potential over a given field range.

    Parameters
    ----------
    potential_func : callable
        A function that returns the value of the potential V(phi).
    derivative_func : callable
        A function that returns the first derivative dV/dphi.
    second_derivative_func : callable
        A function that returns the second derivative d²V/dphi².

    Methods
    -------
    evaluate(phi)
        Evaluates the potential V(phi) at the given value(s) of phi.

    first_derivative(phi)
        Evaluates the first derivative of the potential at phi.

    second_derivative(phi)
        Evaluates the second derivative of the potential at phi.

    from_string(potential_expr_str, param_values={})
        Class method that constructs a PotentialFunction from a string expression and
        a dictionary of parameter values. Uses symbolic differentiation.

    plot_potential(phi_min, phi_max, dpi, num_points=1000, save=False, filename='potential.png')
        Plots the potential V(phi) in the specified range.
    """

    def __init__(self, potential_func, derivative_func, second_derivative_func):
        self.potential_func = potential_func
        self.derivative_func = derivative_func
        self.second_derivative_func = second_derivative_func

    def evaluate(self, phi):
        return self.potential_func(phi)

    def first_derivative(self, phi):
        return self.derivative_func(phi)
    
    def second_derivative(self, phi):
        return self.second_derivative_func(phi)

    @staticmethod
    def from_string(potential_expr_str, param_values={}):
        
        try:
            # Define phi as main symbol
            phi = sp.symbols('phi')

            # Extract the parameter names and define them as symbols            
            param_symbols = {name: sp.symbols(name) for name in param_values.keys()}
            
            #We convert the string to a symbolic expression
            V_expr = sp.sympify(potential_expr_str)
            
            # Validate that the parameters provided are in the expression
            for param in param_symbols.keys():
                if param not in potential_expr_str:
                    raise ValueError(f"The parameter “{param}” is not present in the expression of the potential.")
            
            V_expr = V_expr.subs(param_symbols)
        
            # Derivatives
            dV_expr = sp.diff(V_expr, phi)
            d2V_expr = sp.diff(dV_expr, phi)

           
            V_func = sp.lambdify((phi, *param_symbols.values()), V_expr, 'numpy')
            dV_func = sp.lambdify((phi, *param_symbols.values()), dV_expr, 'numpy')
            d2V_func = sp.lambdify((phi, *param_symbols.values()), d2V_expr, 'numpy')

        except Exception as e:
            raise ValueError(f"Error in interpreting the potential function: {e}")

        # Devolvemos funciones evaluadas con los valores de los parámetros dados
        return PotentialFunction(
            lambda phi: V_func(phi, *param_values.values()),
            lambda phi: dV_func(phi, *param_values.values()),
            lambda phi: d2V_func(phi, *param_values.values())
        )


    def plot_potential(self, phi_min, phi_max, dpi, num_points = 1000, save = False, filename = 'potential.png'):
    
        phi_vals = np.linspace(phi_min, phi_max, num_points)
        V = self.evaluate(phi_vals)

        from .plot_style import style
        style(dpi = dpi)
        plt.plot(phi_vals, V)
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$V(\phi)$')
        plt.tight_layout()

        if save:
            import os

            os.makedirs('Figures', exist_ok = True)
            filepath = os.path.join('figures', filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Figure saved as: {filepath}")
        
        plt.show()        