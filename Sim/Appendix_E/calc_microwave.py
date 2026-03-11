import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Отключаем LaTeX для ускорения
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

# Константы
h = 4.135667662e-15         # eV·s
hbar = h / (2 * np.pi)
k_B_eV_per_K = 8.617333262145e-5  # eV/K

# Параметры материалов (J в мК → eV)
materials = {
    'PMN-PT':    {'J': 0.4e-3 * k_B_eV_per_K, 'dim_S': 1.8, 'T_c': 1.45e-3, 'color': '#1f77b4'},
    'MoS₂':      {'J': 0.35e-3 * k_B_eV_per_K,'dim_S': 1.5, 'T_c': 1.2e-3,  'color': '#ff7f0e'},
    'BGO':       {'J': 0.25e-3 * k_B_eV_per_K,'dim_S': 2.1, 'T_c': 0.9e-3,  'color': '#2ca02c'},
    'PVDF-TiO₂': {'J': 0.42e-3 * k_B_eV_per_K,'dim_S': 1.9, 'T_c': 1.5e-3,  'color': '#d62728'}
}

# 1) Spectral density vs frequency at various T/Tc (PMN-PT)
plt.figure(figsize=(8,6))
freq_ghz = np.linspace(0.1, 5, 500)
for T_ratio in [0.5, 0.8, 1.0, 1.2]:
    m = materials['PMN-PT']
    # ω0 (GHz) = (2π J dim_S / ħ) * (T_ratio)
    omega0_rad = 2*np.pi * m['J'] * m['dim_S'] * T_ratio / hbar
    omega0_ghz = omega0_rad / (2*np.pi * 1e9)
    g = freq_ghz**(m['dim_S']-1) * np.exp(-freq_ghz/omega0_ghz)
    g /= g.max()
    plt.plot(freq_ghz, g, lw=2, label=f'{T_ratio:.1f} Tc')

plt.xlabel('Frequency (GHz)', fontsize=14)
plt.ylabel('Normalized Spectral Density', fontsize=14)
plt.title('Spectral Density vs Frequency at Various T/Tc (PMN-PT)', fontsize=16)
plt.legend(title='T/Tc', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0,5); plt.ylim(0,1.05)
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.tight_layout()
plt.savefig('spectral_density.png', dpi=300, bbox_inches='tight')


# 2) Resonance frequency vs T/Tc for various materials
plt.figure(figsize=(8,6))
T_ratios = np.linspace(0.5, 2.0, 100)
for name, p in materials.items():
    omega_res_ghz = (2*np.pi * p['J'] * p['dim_S'] * T_ratios / hbar) / (2*np.pi*1e9)
    plt.plot(T_ratios, omega_res_ghz, color=p['color'], lw=2, label=name)

plt.xlabel('Temperature Ratio T/Tc', fontsize=14)
plt.ylabel('Resonance Frequency (GHz)', fontsize=14)
plt.title('Resonance Frequency vs T/Tc for Various Materials', fontsize=16)
plt.legend(title='Materials', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0.5,2.0); plt.ylim(0,3.5)
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.tight_layout()
plt.savefig('resonance_frequency.png', dpi=300, bbox_inches='tight')


# 3) Dielectric loss spectrum with Lorentzian dip (PMN-PT at T=0.5Tc)
plt.figure(figsize=(8,6))
freq = np.linspace(0.5, 5, 500)
omega_res = 2.3  # GHz
gamma = 0.2      # GHz
A = 0.08         # Depth of dip
baseline = 0.1
tan_delta = baseline - A * gamma**2 / ((freq - omega_res)**2 + gamma**2)

plt.plot(freq, tan_delta, lw=2, color=materials['PMN-PT']['color'])
plt.annotate('Resonance Dip', xy=(omega_res, baseline - A),
             xytext=(omega_res+0.5, baseline - A + 0.02),
             arrowprops=dict(arrowstyle='->'), fontsize=12)
plt.xlabel('Frequency (GHz)', fontsize=14)
plt.ylabel('Dielectric Loss tan δ', fontsize=14)
plt.title('Predicted Microwave Absorption Spectrum (PMN-PT at T=0.5 Tc)', fontsize=16)
plt.xlim(0.5,5); plt.ylim(0.01,0.11)
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('dielectric_loss.png', dpi=300, bbox_inches='tight')

print("Saved: spectral_density.png, resonance_frequency.png, dielectric_loss.png")
