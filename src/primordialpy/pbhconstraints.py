import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

class PBHConstraints:
    """
    Manages the loading and plotting of observational constraints on PBH abundance.

    Observational bounds are loaded from text files and overlaid on a
    f_PBH vs M_PBH plot. Model predictions can be passed directly as
    arrays, with no dependency on any specific abundance class.
    """

    def __init__(self, data_folder: Optional[str] = None):
        if data_folder is None:
            here = os.path.dirname(os.path.abspath(__file__))
            self.data_folder = os.path.join(here, 'constraints_data')
        else:
            self.data_folder = data_folder

        if not os.path.exists(self.data_folder):
            logger.warning(
                f"Constraints data folder not found: {self.data_folder}. "
                "Ensure 'constraints_data' is inside the 'primordialpy' folder."
            )

        self.bounds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.color: str = 'gray'
        self.alpha_fill: float = 0.15
        self.alpha_edge: float = 0.4

    def __repr__(self) -> str:
        return f"PBHConstraints(data_folder='{self.data_folder}', bounds={list(self.bounds.keys())})"

    def add_bound(self, label: str, filename: str) -> None:
        """Loads an observational constraint from a text file."""
        filepath = os.path.join(self.data_folder, filename)

        if not os.path.exists(filepath):
            logger.warning(f"File '{filename}' not found in {self.data_folder}.")
            return

        try:
            data = np.loadtxt(filepath)
            sort_idx = np.argsort(data[:, 0])
            m = data[sort_idx, 0]
            f = data[sort_idx, 1]
            self.bounds[label] = (m, f)
            logger.info(f"Loaded constraint: {label}")

        except Exception as e:
            logger.error(f"Failed to read '{filename}': {e}")

    def plot(self, ax: Optional[Axes] = None, xlims: Optional[Tuple[float, float]] = None) -> Axes:
        """Plots all loaded observational constraints as shaded regions."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        if xlims is None:
            xlims = (1e-18, 1e2)

        for label, (m, f) in self.bounds.items():
            mask = (m >= xlims[0] * 0.1) & (m <= xlims[1] * 10)
            if not np.any(mask):
                continue

            m_plot = m[mask]
            f_plot = f[mask]

            ax.fill_between(m_plot, f_plot, 100.0, color=self.color, alpha=self.alpha_fill, linewidth=0)
            ax.plot(m_plot, f_plot, color=self.color, alpha=self.alpha_edge, linewidth=0.8)
            self._add_smart_label(ax, label, m_plot, f_plot)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$M_{\rm PBH}\ [M_\odot]$', fontsize=14)
        ax.set_ylabel(r'$f_{\rm PBH}(M)$', fontsize=14)

        return ax

    def _add_smart_label(self, ax: Axes, label: str, m: np.ndarray, f: np.ndarray) -> None:
        """Places a label near the minimum of the constraint curve."""
        idx_min = np.argmin(f)
        ax.text(m[idx_min], f[idx_min] * 2.5, label, color='dimgray', fontsize=9, ha='center', va='bottom', rotation=0)

    def plot_constraints(self, m_pbh: np.ndarray, f_pbh: np.ndarray, label: Optional[str] = None, filename: str = 'constraints_check.png', path: str = 'Figures', **kwargs) -> Axes:
        """Plots observational constraints together with a model prediction."""
        try:
            from .plot_style import style
            style()
        except ImportError:
            pass

        fig, ax = plt.subplots(figsize=(8, 6))

        xlims = (1e-18, 1e2)
        self.plot(ax, xlims=xlims)

        plot_kwargs = dict(color='firebrick', linewidth=1.5)
        plot_kwargs.update(kwargs)
        ax.loglog(m_pbh, f_pbh, label=label, **plot_kwargs)

        ax.axhline(1, color='gray', linestyle='--', alpha=0.5, linewidth=1, label=r'$f_{\rm PBH} = 1$')

        ax.set_xlim(1e-18, 1e3)
        ax.set_ylim(1e-10, 2.0)
        ax.legend(loc='lower left', frameon=False)
        ax.set_title("Observational Constraints", fontsize=16)

        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Figure saved to: {save_path}")

        return ax