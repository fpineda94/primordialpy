import numpy as np
import matplotlib.pyplot as plt
import os

from primordialpy.pbhabundance import PBHAbundance


class PBHConstraints:
    """
    Manage the loading and plotting of observational data.
    """
    def __init__(self, data_folder=None):
           
            if data_folder is None:
                here = os.path.dirname(os.path.abspath(__file__))
                
                self.data_folder = os.path.join(here, 'constraints_data')
            else:
                self.data_folder = data_folder

            if not os.path.exists(self.data_folder):
                print(f"[Warning] I couldn't find the data folder in: {self.data_folder}")
                print("Ensure that ‘constraints_data’ is inside the folder. 'primordialpy'.")

            self.bounds = {}
            self.color = 'gray' 
            self.alpha_fill = 0.15
            self.alpha_edge = 0.4

    def add_bound(self, label, filename):
     
        filepath = os.path.join(self.data_folder, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: I couldn't find '{filename}' in {self.data_folder}")
            return

        try:
            data = np.loadtxt(filepath)
            
            sort_idx = np.argsort(data[:, 0])
            m = data[sort_idx, 0]
            f = data[sort_idx, 1]
            
            self.bounds[label] = (m, f)
            print(f"  [OK] Loaded data: {label}")
            
        except Exception as e:
            print(f"  [ERROR] Failure to read {filename}: {e}")

    def plot(self, ax=None, xlims= None):
      
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        if xlims is None:
            xlims = (1e-18, 1e2)

        for label, (m, f) in self.bounds.items():
            
            mask = (m >= xlims[0] * 0.1) & (m <= xlims[1] * 10)
            if not np.any(mask): 
                continue
                
            m_plot = m[mask]
            f_plot = f[mask]

            ax.fill_between(m_plot, f_plot, 100.0, 
                            color=self.color, alpha=self.alpha_fill, linewidth=0)
            
            ax.plot(m_plot, f_plot, 
                    color=self.color, alpha=self.alpha_edge, linewidth=0.8)

            self._add_smart_label(ax, label, m_plot, f_plot)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$M_{\rm PBH} [M_\odot]$', fontsize=14)
        ax.set_ylabel(r'$f_{\rm PBH} (M)$', fontsize=14)
        
        return ax

    def _add_smart_label(self, ax, label, m, f):
        idx_min = np.argmin(f)
        x_pos = m[idx_min]
        y_pos = f[idx_min]
        
        ax.text(x_pos, y_pos * 2.5, label, 
                color='dimgray', fontsize=9, 
                ha='center', va='bottom', rotation=0)
        

    def plot_constraints(self, model: PBHAbundance, label = None, filename='constraints_check.png'):
     
        try:
            from .plot_style import style
            style()
        except ImportError:
            pass

        fig, ax = plt.subplots(figsize=(8, 6))

        m_model, _ = model.fPBH(save=False)
        m_min, m_max = np.min(m_model), np.max(m_model)
        
        xlims = (1e-18, 1e2)
        self.plot(ax, xlims=xlims)

        model.plot_abundance(
            ax=ax, 
            color='firebrick', 
            linewidth=1.5, 
            label=label,
            ylim_bottom=1e-10 
        )
        ax.set_xlim(1e-18, 1e3)
        ax.set_ylim(1e-10, 2.0)
        
        ax.legend(loc='lower left', frameon=False)
        ax.set_title("Observational Constraints", fontsize=16)
        
        save_path = os.path.join('Figures', filename)
        os.makedirs('Figures', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")