import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from io import BytesIO
import base64
from streamlit_bokeh_events import streamlit_bokeh_events
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy import signal
import pywt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from concurrent.futures import ProcessPoolExecutor
import plotly.graph_objects as go
import lasio
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import time
import sys
import traceback
from functools import wraps

# Initialize session state
if 'logs' not in st.session_state:
    st.session_state.logs = None
if 'mc_results' not in st.session_state:
    st.session_state.mc_results = None
if 'wavelet' not in st.session_state:
    st.session_state.wavelet = None
if 'syn_gathers' not in st.session_state:
    st.session_state.syn_gathers = None
if 'freq_results' not in st.session_state:
    st.session_state.freq_results = {}

# Error handling decorator
def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            st.error(traceback.format_exc())
            return None
    return wrapper

# Import rockphypy with error handling
try:
    from rockphypy import QI, GM, Fluid
    rockphypy_available = True
except ImportError:
    rockphypy_available = False
    st.warning("rockphypy package not available - some models will be disabled")

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced Rock Physics & AVO Modeling with Wedge Modeling")

# Title and description
st.title("Enhanced Rock Physics & AVO Modeling Tool with Wedge Modeling")
st.markdown("""
This app performs advanced rock physics modeling, AVO analysis, and wedge modeling with multiple models, 
visualization options, uncertainty analysis, sonic log prediction, and seismic inversion feasibility assessment.
""")

# Available colormaps for seismic displays
seismic_colormaps = ['seismic', 'RdBu', 'bwr', 'coolwarm', 'viridis', 'plasma']

# ==============================================
# Enhanced Rock Physics Model Functions
# ==============================================

@handle_errors
def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
    """Gassmann's Fluid Substitution"""
    vp1 = vp1/1000.  # Convert m/s to km/s
    vs1 = vs1/1000.
    mu1 = rho1 * vs1**2
    k_s1 = rho1 * vp1**2 - (4./3.) * mu1

    # Dry rock bulk modulus (Gassmann's equation)
    kdry = (k_s1*((phi*k0)/k_f1 + 1 - phi) - k0) / \
           ((phi*k0)/k_f1 + (k_s1/k0) - 1 - phi)

    # Apply Gassmann to get new fluid properties
    k_s2 = kdry + (1 - (kdry/k0))**2 / \
           ((phi/k_f2) + ((1-phi)/k0) - (kdry/k0**2))
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    mu2 = mu1  # Shear modulus unaffected by fluid change
    vp2 = np.sqrt((k_s2 + (4./3)*mu2) / rho2)
    vs2 = np.sqrt(mu2 / rho2)

    return vp2*1000, vs2*1000, rho2, k_s2

@handle_errors
def critical_porosity_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, phi_c):
    """Critical Porosity Model (Nur et al.)"""
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1*vs1**2.
    k_s1 = rho1*vp1**2 - (4./3.)*mu1
    
    # Modified dry rock modulus for critical porosity
    kdry = k0 * (1 - phi/phi_c)
    mudry = mu0 * (1 - phi/phi_c)
    
    # Gassmann substitution
    k_s2 = kdry + (1-(kdry/k0))**2/((phi/k_f2)+((1-phi)/k0)-(kdry/k0**2))
    rho2 = rho1-phi*rho_f1+phi*rho_f2
    mu2 = mudry
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2)
    vs2 = np.sqrt(mu2/rho2)
    
    return vp2*1000, vs2*1000, rho2, k_s2

@handle_errors
def hertz_mindlin_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P):
    """Hertz-Mindlin contact theory model"""
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1*vs1**2.
    k_s1 = rho1*vp1**2 - (4./3.)*mu1
    
    # Hertz-Mindlin dry rock moduli
    PR0 = (3*k0 - 2*mu0)/(6*k0 + 2*mu0)  # Poisson's ratio
    kdry = (Cn**2 * (1 - phi)**2 * P * mu0**2 / (18 * np.pi**2 * (1 - PR0)**2))**(1/3)
    mudry = ((2 + 3*PR0 - PR0**2)/(5*(2 - PR0))) * (
        (3*Cn**2 * (1 - phi)**2 * P * mu0**2)/(2 * np.pi**2 * (1 - PR0)**2)
    )**(1/3)
    
    # Gassmann substitution
    k_s2 = kdry + (1-(kdry/k0))**2/((phi/k_f2)+((1-phi)/k0)-(kdry/k0**2))
    rho2 = rho1-phi*rho_f1+phi*rho_f2
    mu2 = mudry
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2)
    vs2 = np.sqrt(mu2/rho2)
    
    return vp2*1000, vs2*1000, rho2, k_s2

@handle_errors
def dvorkin_nur_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn=9, P=10, phi_c=0.4):
    """Dvorkin-Nur Soft Sand model for unconsolidated sands"""
    vp1 = vp1/1000.  # Convert to km/s
    vs1 = vs1/1000.
    
    # Hertz-Mindlin for dry rock moduli at critical porosity
    PR0 = (3*k0 - 2*mu0)/(6*k0 + 2*mu0)  # Poisson's ratio
    
    # Dry rock moduli at critical porosity
    k_hm = (Cn**2 * (1-phi_c)**2 * P * mu0**2 / (18 * np.pi**2 * (1-PR0)**2))**(1/3)
    mu_hm = ((2 + 3*PR0 - PR0**2)/(5*(2-PR0))) * (
        (3*Cn**2 * (1-phi_c)**2 * P * mu0**2)/(2*np.pi**2*(1-PR0)**2)
    )**(1/3)
    
    # Modified Hashin-Shtrikman lower bound for dry rock
    k_dry = (phi/phi_c)/(k_hm + (4/3)*mu_hm) + (1 - phi/phi_c)/(k0 + (4/3)*mu_hm)
    k_dry = 1/k_dry - (4/3)*mu_hm
    k_dry = np.maximum(k_dry, 0)  # Ensure positive values
    
    mu_dry = (phi/phi_c)/(mu_hm + (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm))) + \
             (1 - phi/phi_c)/(mu0 + (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm)))
    mu_dry = 1/mu_dry - (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm))
    mu_dry = np.maximum(mu_dry, 0)
    
    # Gassmann fluid substitution
    k_sat = k_dry + (1 - (k_dry/k0))**2 / ((phi/k_f2) + ((1-phi)/k0) - (k_dry/k0**2))
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    vp2 = np.sqrt((k_sat + (4/3)*mu_dry)/rho2) * 1000  # Convert back to m/s
    vs2 = np.sqrt(mu_dry/rho2) * 1000
    
    return vp2, vs2, rho2, k_sat

@handle_errors
def raymer_hunt_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
    """Raymer-Hunt-Gardner empirical model"""
    # Empirical relationships for dry rock
    vp_dry = (1 - phi)**2 * np.sqrt(k0/rho1) + phi * np.sqrt(k_f1/rho_f1)
    vp_dry = vp_dry * 1000  # Convert to m/s
    
    # For saturated rock
    vp_sat = (1 - phi)**2 * np.sqrt(k0/rho1) + phi * np.sqrt(k_f2/rho_f2)
    vp_sat = vp_sat * 1000
    
    # VS is less affected by fluids (use empirical relationship)
    vs_sat = vs1 * (1 - 1.5*phi)  # Simple porosity correction
    
    # Density calculation
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    
    return vp_sat, vs_sat, rho2, None  # No bulk modulus returned in this model

@handle_errors
def xu_payne_model(vp1, vs1, rho1, phi, vsh, c=1.0, k0=37.0, mu0=44.0, k_sh=15.0, mu_sh=5.0):
    """Xu-Payne laminated sand-shale model"""
    # Mineral properties
    k_qz = 37.0
    mu_qz = 44.0
    
    # Calculate dry rock moduli using Hashin-Shtrikman bounds
    f_sh = vsh / (vsh + (1 - vsh) * c)
    k_dry = ((1 - f_sh) / (k_qz + (4/3)*mu_qz) + f_sh / (k_sh + (4/3)*mu_qz))**(-1) - (4/3)*mu_qz
    mu_dry = ((1 - f_sh) / (mu_qz + mu_qz/6*((9*k_qz + 8*mu_qz)/(k_qz + 2*mu_qz))) + 
              f_sh / (mu_sh + mu_qz/6*((9*k_qz + 8*mu_qz)/(k_qz + 2*mu_qz))))**(-1) - \
             mu_qz/6*((9*k_qz + 8*mu_qz)/(k_qz + 2*mu_qz))
    
    # Gassmann fluid substitution (using water properties as default)
    k_fl = 2.8  # Brine bulk modulus in GPa
    k_sat = k_dry + (1 - k_dry/k0)**2 / (phi/k_fl + (1-phi)/k0 - k_dry/k0**2)
    
    # Density calculation
    rho2 = rho1 * (1 - phi) + phi * 1.0  # Assuming water saturation
    
    # Calculate velocities
    vp2 = np.sqrt((k_sat + (4/3)*mu_dry)/rho2) * 1000  # Convert to m/s
    vs2 = np.sqrt(mu_dry/rho2) * 1000
    
    return vp2, vs2, rho2, k_sat

@handle_errors
def greenberg_castagna(vp, vs, rho, phi, sw, lithology='sandstone'):
    """Greenberg-Castagna empirical Vp-Vs relationships"""
    # Empirical coefficients for different lithologies
    coefficients = {
        'sandstone': {'a': 0.8042, 'b': -0.8559},
        'shale': {'a': 0.7697, 'b': -0.8674},
        'carbonate': {'a': 0.8535, 'b': -1.1375},
        'dolomite': {'a': 0.7825, 'b': -0.5529}
    }
    
    # Get coefficients based on lithology
    coeff = coefficients.get(lithology, coefficients['sandstone'])
    
    # Predict Vs from Vp (convert km/s to m/s)
    vp_km = vp / 1000
    vs_pred_km = coeff['a'] * vp_km + coeff['b']
    vs_pred = vs_pred_km * 1000
    
    # Adjust for porosity and saturation (simplified)
    vs_corr = vs_pred * (1 - 1.5*phi) * (1 - 0.5*sw)
    
    return vp, vs_corr, rho, None

# ==============================================
# Enhanced AVO and Seismic Modeling Functions
# ==============================================

@st.cache_data
def ricker_wavelet(frequency, length=0.128, dt=0.001):
    """Generate a Ricker wavelet with caching"""
    t = np.linspace(-length/2, length/2, int(length/dt))
    y = (1 - 2*(np.pi**2)*(frequency**2)*(t**2)) * np.exp(-(np.pi**2)*(frequency**2)*(t**2))
    return t, y

@handle_errors
def smith_gidlow(vp1, vp2, vs1, vs2, rho1, rho2):
    """Calculate Smith-Gidlow AVO attributes (intercept, gradient)"""
    # Calculate reflectivities
    rp = 0.5 * (vp2 - vp1) / (vp2 + vp1) + 0.5 * (rho2 - rho1) / (rho2 + rho1)
    rs = 0.5 * (vs2 - vs1) / (vs2 + vs1) + 0.5 * (rho2 - rho1) / (rho2 + rho1)
    
    # Smith-Gidlow coefficients
    intercept = rp
    gradient = rp - 2 * rs
    fluid_factor = rp + 1.16 * (vp1/vs1) * rs
    
    return intercept, gradient, fluid_factor

@handle_errors
def calculate_reflection_coefficients(vp1, vp2, vs1, vs2, rho1, rho2, angle):
    """Calculate PP reflection coefficients using Aki-Richards approximation"""
    theta = np.radians(angle)
    vp_avg = (vp1 + vp2)/2
    vs_avg = (vs1 + vs2)/2
    rho_avg = (rho1 + rho2)/2
    
    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    
    a = 0.5 * (1 + np.tan(theta)**2)
    b = -4 * (vs_avg**2/vp_avg**2) * np.sin(theta)**2
    c = 0.5 * (1 - 4 * (vs_avg**2/vp_avg**2) * np.sin(theta)**2)
    
    rc = a*(dvp/vp_avg) + b*(dvs/vs_avg) + c*(drho/rho_avg)
    return rc

@handle_errors
def fit_avo_curve(angles, rc_values):
    """Fit a line to AVO response to get intercept and gradient"""
    def linear_func(x, intercept, gradient):
        return intercept + gradient * np.sin(np.radians(x))**2
    
    try:
        popt, pcov = curve_fit(linear_func, angles, rc_values)
        intercept, gradient = popt
        return intercept, gradient, np.sqrt(np.diag(pcov))
    except:
        return np.nan, np.nan, (np.nan, np.nan)

# ==============================================
# Enhanced Wedge Modeling Functions
# ==============================================

@handle_errors
def calc_rc(vp_mod, rho_mod):
    """Calculate reflection coefficients"""
    nlayers = len(vp_mod)
    nint = nlayers - 1
    rc_int = []
    for i in range(0, nint):
        buf1 = vp_mod[i+1]*rho_mod[i+1]-vp_mod[i]*rho_mod[i]
        buf2 = vp_mod[i+1]*rho_mod[i+1]+vp_mod[i]*rho_mod[i]
        buf3 = buf1/buf2
        rc_int.append(buf3)
    return rc_int

@handle_errors
def calc_times(z_int, vp_mod):
    """Calculate two-way times to interfaces"""
    nlayers = len(vp_mod)
    nint = nlayers - 1
    t_int = []
    for i in range(0, nint):
        if i == 0:
            tbuf = z_int[i]/vp_mod[i]
            t_int.append(tbuf)
        else:
            zdiff = z_int[i]-z_int[i-1]
            tbuf = 2*zdiff/vp_mod[i] + t_int[i-1]
            t_int.append(tbuf)
    return t_int

@handle_errors
def digitize_model(rc_int, t_int, t):
    """Digitize model for convolution"""
    nlayers = len(rc_int)
    nint = nlayers - 1
    nsamp = len(t)
    rc = list(np.zeros(nsamp,dtype='float'))
    lyr = 0
    
    for i in range(0, nsamp):
        if t[i] >= t_int[lyr]:
            rc[i] = rc_int[lyr]
            lyr = lyr + 1    
        if lyr > nint:
            break
    return rc

@handle_errors
def plot_vawig(axhdl, data, t, excursion, highlight=None):
    """Plot variable area wiggle traces"""
    [ntrc, nsamp] = data.shape
    t = np.hstack([0, t, t.max()])
    
    for i in range(0, ntrc):
        tbuf = excursion * data[i] / np.max(np.abs(data)) + i
        tbuf = np.hstack([i, tbuf, i])
            
        if i==highlight:
            lw = 2
        else:
            lw = 0.5

        axhdl.plot(tbuf, t, color='black', linewidth=lw)
        plt.fill_betweenx(t, tbuf, i, where=tbuf>i, facecolor=[0.6,0.6,1.0], linewidth=0)
        plt.fill_betweenx(t, tbuf, i, where=tbuf<i, facecolor=[1.0,0.7,0.7], linewidth=0)
    
    axhdl.set_xlim((-excursion, ntrc+excursion))
    axhdl.xaxis.tick_top()
    axhdl.xaxis.set_label_position('top')
    axhdl.invert_yaxis()

# ==============================================
# Enhanced Frequency Analysis Functions
# ==============================================

@handle_errors
def analyze_frequency_content(syn_gather, t_samp, wavelet_freq):
    """Perform comprehensive frequency analysis on synthetic gather"""
    results = {}
    
    # Time range filtering
    time_mask = (t_samp >= 0.15) & (t_samp <= 0.25)
    t_samp_filtered = t_samp[time_mask]
    
    # FFT Analysis
    n = syn_gather.shape[1]
    dt = t_samp[1] - t_samp[0]
    freqs = np.fft.rfftfreq(n, dt)
    
    # Initialize array to store frequency spectra
    freq_spectra = np.zeros((len(t_samp_filtered), len(freqs)))
    
    # Calculate FFT for each time sample
    for i, t in enumerate(t_samp_filtered):
        time_idx = np.where(t_samp == t)[0][0]
        time_slice = syn_gather[:, time_idx]
        
        # Apply windowing
        window = np.hanning(len(time_slice))
        windowed_signal = time_slice * window
        
        # Compute FFT
        spectrum = np.abs(np.fft.rfft(windowed_signal))
        min_len = min(len(spectrum), len(freqs))
        freq_spectra[i, :min_len] = spectrum[:min_len]
    
    # Normalize
    if np.max(freq_spectra) > 0:
        freq_spectra = freq_spectra / np.max(freq_spectra)
    
    results['fft'] = {
        'freqs': freqs,
        'spectra': freq_spectra,
        'time_samples': t_samp_filtered
    }
    
    # CWT Analysis
    scales = np.arange(1, 51)  # 1-50 scales
    wavelet = 'morl'
    cwt_freqs = pywt.scale2frequency(wavelet, scales) / dt
    
    # Initialize array to store CWT magnitudes
    cwt_magnitudes = np.zeros((len(t_samp_filtered), len(scales)))
    
    for i, t in enumerate(t_samp_filtered):
        time_idx = np.where(t_samp == t)[0][0]
        trace = syn_gather[:, time_idx]
        
        if len(trace) > 0:
            coefficients, _ = pywt.cwt(trace, scales, wavelet, sampling_period=dt)
            cwt_magnitudes[i, :] = np.sum(np.abs(coefficients), axis=1)
    
    if cwt_magnitudes.size > 0:
        results['cwt'] = {
            'scales': scales,
            'freqs': cwt_freqs,
            'magnitudes': cwt_magnitudes,
            'time_samples': t_samp_filtered,
            'wavelet': wavelet
        }
    
    return results

# ==============================================
# Helper Functions
# ==============================================

@handle_errors
def monte_carlo_iteration(params):
    """Single iteration of Monte Carlo simulation"""
    # Perturb input parameters with normal distribution
    perturbed_params = {}
    for param, (mean, std) in params.items():
        perturbed_params[param] = np.random.normal(mean, std) if std > 0 else mean
    
    # Apply model with perturbed parameters
    vp, vs, rho, _ = model_func(**perturbed_params)
    
    # Calculate derived properties
    ip = vp * rho
    vpvs = vp / vs
    
    # Calculate Smith-Gidlow attributes (using mean values for simplicity)
    vp_upper = logs.VP.mean()
    vs_upper = logs.VS.mean()
    rho_upper = logs.RHO.mean()
    intercept, gradient, fluid_factor = smith_gidlow(vp_upper, vp, vs_upper, vs, rho_upper, rho)
    
    return {
        'VP': vp, 'VS': vs, 'RHO': rho,
        'IP': ip, 'VPVS': vpvs,
        'Intercept': intercept, 'Gradient': gradient, 'Fluid_Factor': fluid_factor
    }

@handle_errors
def parallel_monte_carlo(logs, model_func, params, iterations=100):
    """Perform parallel Monte Carlo simulation for uncertainty analysis"""
    results = {
        'VP': [], 'VS': [], 'RHO': [], 
        'IP': [], 'VPVS': [], 'Intercept': [], 
        'Gradient': [], 'Fluid_Factor': []
    }
    
    # Prepare arguments for parallel processing
    args = [(i, params) for i in range(iterations)]
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(0, iterations, 10):  # Process in chunks of 10
            chunk = args[i:i+10]
            futures.append(executor.submit(process_monte_carlo_chunk, logs, model_func, chunk))
            
            # Update progress
            progress = min((i + 10) / iterations, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing {int(progress*100)}% complete...")
        
        # Collect results
        for future in futures:
            chunk_results = future.result()
            for key in results:
                results[key].extend(chunk_results[key])
    
    status_text.text("Monte Carlo simulation complete!")
    return results

@handle_errors
def process_monte_carlo_chunk(logs, model_func, args):
    """Process a chunk of Monte Carlo iterations"""
    chunk_results = {
        'VP': [], 'VS': [], 'RHO': [], 
        'IP': [], 'VPVS': [], 'Intercept': [], 
        'Gradient': [], 'Fluid_Factor': []
    }
    
    for i, params in args:
        result = monte_carlo_iteration(params)
        for key in chunk_results:
            chunk_results[key].append(result[key])
    
    return chunk_results

@handle_errors
def create_interactive_crossplot(logs, depth_range=None):
    """Create interactive crossplot with Bokeh"""
    if depth_range:
        logs = logs[(logs['DEPTH'] >= depth_range[0]) & (logs['DEPTH'] <= depth_range[1])]
    
    # Convert numeric factors to strings
    logs['LFC_MIX'] = logs['LFC_MIX'].astype(str)
    
    source = ColumnDataSource(logs)
    palette = Category10[10]
    
    p = figure(tools="pan,wheel_zoom,box_zoom,reset,hover,save",
               title="Interactive Crossplot")
    
    # Use string factors
    factors = sorted(logs['LFC_MIX'].unique().astype(str))
    cmap = factor_cmap('LFC_MIX', palette=palette, factors=factors)
    
    p.scatter('IP_FRMMIX', 'VPVS_FRMMIX', size=8, source=source,
              color=cmap, alpha=0.6, legend_field='LFC_MIX')
    
    p.add_tools(HoverTool(
        tooltips=[
            ("Depth", "@DEPTH"),
            ("IP", "@IP_FRMMIX"),
            ("Vp/Vs", "@VPVS_FRMMIX"),
            ("Class", "@LFC_MIX")
        ]
    ))
    
    return p

@handle_errors
def create_interactive_3d_crossplot(logs, x_col='IP', y_col='VPVS', z_col='RHO', color_col='LFC_B'):
    """Create interactive 3D crossplot with Plotly"""
    # Define color mapping
    color_map = {
        0: 'gray',    # Undefined
        1: 'blue',    # Brine
        2: 'green',   # Oil
        3: 'red',     # Gas
        4: 'magenta', # Mixed
        5: 'brown'    # Shale
    }
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for each class
    for class_val, color in color_map.items():
        mask = logs[color_col] == class_val
        if mask.any():
            fig.add_trace(go.Scatter3d(
                x=logs.loc[mask, x_col],
                y=logs.loc[mask, y_col],
                z=logs.loc[mask, z_col],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color,
                    opacity=0.7,
                    line=dict(width=0)
                ),
                name=f'Class {class_val}',
                hovertemplate=
                    f"<b>{x_col}</b>: %{{x:.2f}}<br>" +
                    f"<b>{y_col}</b>: %{{y:.2f}}<br>" +
                    f"<b>{z_col}</b>: %{{z:.2f}}<br>" +
                    "<extra></extra>"
            ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            xaxis=dict(backgroundcolor="rgb(200, 200, 230)"),
            yaxis=dict(backgroundcolor="rgb(230, 200, 230)"),
            zaxis=dict(backgroundcolor="rgb(230, 230, 200)"),
            bgcolor="rgb(255, 255, 255)",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.1,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

@handle_errors
def predict_sonic_logs(logs, features, target_vp='VP', target_vs='VS', model_type='Random Forest'):
    """Predict VP and VS using selected machine learning model"""
    # Select features and target
    X = logs[features].dropna()
    y_vp = logs.loc[X.index, target_vp]
    y_vs = logs.loc[X.index, target_vs]
    
    # Train-test split
    X_train, X_test, y_vp_train, y_vp_test = train_test_split(
        X, y_vp, test_size=0.2, random_state=42)
    _, _, y_vs_train, y_vs_test = train_test_split(
        X, y_vs, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select model
    if model_type == 'Random Forest':
        vp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        vs_model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'XGBoost':
        vp_model = XGBRegressor(n_estimators=100, random_state=42)
        vs_model = XGBRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Gaussian Process':
        vp_model = GaussianProcessRegressor(random_state=42)
        vs_model = GaussianProcessRegressor(random_state=42)
    elif model_type == 'Neural Network':
        vp_model = MLPRegressor(hidden_layer_sizes=(100,50), random_state=42, max_iter=500)
        vs_model = MLPRegressor(hidden_layer_sizes=(100,50), random_state=42, max_iter=500)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train models
    vp_model.fit(X_train_scaled, y_vp_train)
    vs_model.fit(X_train_scaled, y_vs_train)
    
    # Predictions
    vp_pred = vp_model.predict(X_test_scaled)
    vs_pred = vs_model.predict(X_test_scaled)
    
    # Metrics
    vp_r2 = r2_score(y_vp_test, vp_pred)
    vs_r2 = r2_score(y_vs_test, vs_pred)
    vp_rmse = np.sqrt(mean_squared_error(y_vp_test, vp_pred))
    vs_rmse = np.sqrt(mean_squared_error(y_vs_test, vs_pred))
    
    # Feature importance
    if hasattr(vp_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': features,
            'VP_Importance': vp_model.feature_importances_,
            'VS_Importance': vs_model.feature_importances_
        })
    else:
        feature_importance = None
    
    return {
        'vp_model': vp_model,
        'vs_model': vs_model,
        'scaler': scaler,
        'vp_r2': vp_r2,
        'vs_r2': vs_r2,
        'vp_rmse': vp_rmse,
        'vs_rmse': vs_rmse,
        'feature_importance': feature_importance,
        'features': features
    }

@handle_errors
def seismic_inversion_feasibility(logs, wavelet_freq):
    """Analyze seismic inversion feasibility"""
    results = {}
    
    # 1. Bandwidth analysis
    nyquist = 0.5 / (logs.DEPTH.diff().mean()/logs.VP.mean())
    wavelet_bandwidth = wavelet_freq * 2  # Approximate
    results['bandwidth'] = {
        'nyquist': nyquist,
        'wavelet': wavelet_bandwidth,
        'feasible': wavelet_bandwidth < nyquist
    }
    
    # 2. AI/VP/VS correlation matrix
    corr_matrix = logs[['IP', 'VP', 'VS', 'RHO', 'VSH', 'PHI']].corr()
    
    # 3. Sensitivity analysis
    sensitivity = {
        'VP_to_PHI': np.corrcoef(logs.VP, logs.PHI)[0,1],
        'VS_to_VSH': np.corrcoef(logs.VS, logs.VSH)[0,1],
        'IP_to_PHI': np.corrcoef(logs.IP, logs.PHI)[0,1]
    }
    
    # 4. Synthetic seismogram resolution
    tuning_thickness = logs.VP.mean() / (4 * wavelet_freq)
    
    return {
        'bandwidth': results['bandwidth'],
        'correlation_matrix': corr_matrix,
        'sensitivity': sensitivity,
        'tuning_thickness': tuning_thickness,
        'resolution': {
            'vertical': tuning_thickness,
            'horizontal': tuning_thickness * 2  # Approximate
        }
    }

@handle_errors
def get_table_download_link(df, filename="results.csv"):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

@handle_errors
def plot_rpt_with_gassmann(title, fluid='gas', 
                         rho_qz=2.65, k_qz=37.0, mu_qz=44.0,
                         rho_sh=2.81, k_sh=15.0, mu_sh=5.0,
                         rho_b=1.09, k_b=2.8,
                         rho_o=0.78, k_o=0.94,
                         rho_g=0.25, k_g=0.06,
                         phi_c=0.4, Cn=9, sigma=20, f=0.5,
                         sw=0.8, so=0.15, sg=0.05,
                         sand_cutoff=0.12):
    """Enhanced RPT plotting with data points"""
    if not rockphypy_available:
        st.error("rockphypy package not available")
        return
    
    plt.figure(figsize=(10, 8))
    
    # 1. Calculate dry rock properties using rockphypy
    phi = np.linspace(0.01, phi_c, 20)  # Porosity range
    
    # Calculate dry rock moduli
    if "Soft Sand" in title:
        Kdry, Gdry = GM.softsand(k_qz, mu_qz, phi, phi_c, Cn, sigma, f=f)
    else:
        Kdry, Gdry = GM.stiffsand(k_qz, mu_qz, phi, phi_c, Cn, sigma, f=f)
    
    # 2. Create colormap for facies
    ccc = ['#B3B3B3','blue','green','red','magenta','#996633']
    cmap_facies = colors.ListedColormap(ccc[0:6], 'indexed')
    
    # 3. Plot RPT background
    if fluid == 'gas':
        QI.plot_rpt(Kdry, Gdry, k_qz, rho_qz, k_b, rho_b, k_g, rho_g, phi, np.linspace(0,1,5))
    elif fluid == 'oil':
        QI.plot_rpt(Kdry, Gdry, k_qz, rho_qz, k_b, rho_b, k_o, rho_o, phi, np.linspace(0,1,5))
    else:  # mixed
        K_mix = (k_o * so + k_g * sg) / (so + sg + 1e-10)
        D_mix = (rho_o * so + rho_g * sg) / (so + sg + 1e-10)
        QI.plot_rpt(Kdry, Gdry, k_qz, rho_qz, k_b, rho_b, K_mix, D_mix, phi, np.linspace(0,1,5))
    
    # 4. Plot actual data points if logs exist
    if 'logs' in st.session_state and all(col in st.session_state.logs.columns for col in ['K_DRY', 'G_DRY', 'LFC_MIX']):
        # Get samples from selected depth range
        depth_mask = (st.session_state.logs['DEPTH'] >= ztop) & (st.session_state.logs['DEPTH'] <= zbot)
        k_dry = st.session_state.logs.loc[depth_mask, 'K_DRY'] / 1e9  # Convert to GPa
        g_dry = st.session_state.logs.loc[depth_mask, 'G_DRY'] / 1e9
        lfc = st.session_state.logs.loc[depth_mask, 'LFC_MIX']
        
        plt.scatter(k_dry, g_dry, c=lfc, 
                   cmap=cmap_facies, s=50, 
                   edgecolors='k', alpha=0.7,
                   vmin=0, vmax=5)
        
        # Add colorbar for facies
        cbar = plt.colorbar()
        cbar.set_label('Fluid Type')
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.set_ticklabels(['Undef','Brine','Oil','Gas','Mixed','Shale'])
    
    plt.title(f"{title} - {fluid.capitalize()} Case")
    plt.xlabel("P-Impedance (km/s*g/cc)")
    plt.ylabel("Vp/Vs Ratio")
    
    # Save and display
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close()

# ==============================================
# Data Processing Function
# ==============================================

@handle_errors
def process_data(uploaded_file, model_choice, include_uncertainty=False, mc_iterations=100, **kwargs):
    """Process uploaded data with selected rock physics model"""
    # Read and validate data
    if isinstance(uploaded_file, str):
        logs = pd.read_csv(uploaded_file)
    else:
        # Handle both CSV and LAS files
        uploaded_file.seek(0)
        if uploaded_file.name.endswith('.las'):
            try:
                las = lasio.read(uploaded_file)
                logs = las.df()
                logs['DEPTH'] = logs.index
            except Exception as e:
                st.error(f"Failed to read LAS file: {str(e)}")
                return None, None
        else:
            try:
                logs = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read CSV file: {str(e)}")
                return None, None
    
    required_columns = {'DEPTH', 'VP', 'VS', 'RHO', 'VSH', 'SW', 'PHI'}
    if not required_columns.issubset(logs.columns):
        missing = required_columns - set(logs.columns)
        st.error(f"Missing required columns: {missing}")
        return None, None
    
    # [Rest of data processing...]
    
    return logs, mc_results

# ==============================================
# Main Application
# ==============================================

def main():
    try:
        # Sidebar for input parameters
        with st.sidebar:
            st.header("Model Configuration")
            
            # Rock physics model selection
            model_options = [
                "Gassmann's Fluid Substitution", 
                "Critical Porosity Model (Nur)", 
                "Contact Theory (Hertz-Mindlin)",
                "Dvorkin-Nur Soft Sand Model",
                "Raymer-Hunt-Gardner Model",
                "Xu-Payne Laminated Model",
                "Greenberg-Castagna Empirical"
            ]
            
            if rockphypy_available:
                model_options.extend([
                    "Soft Sand RPT (rockphypy)",
                    "Stiff Sand RPT (rockphypy)"
                ])
            
            model_choice = st.selectbox("Rock Physics Model", model_options, index=0,
                                      help="Select the rock physics model to use for analysis")
            
            # [Rest of sidebar controls...]
            
            # File upload
            with st.expander("Data Input"):
                uploaded_file = st.file_uploader("Upload CSV or LAS file", type=["csv", "las"], key="file_uploader")
        
        # Main content area
        if uploaded_file is not None:
            # Process data with selected model
            logs, mc_results = process_data(
                uploaded_file, 
                model_choice,
                include_uncertainty=include_uncertainty,
                mc_iterations=mc_iterations,
                # [Other parameters...]
            )
            
            if logs is None:
                st.error("Data processing failed - check your input data")
                return
            
            # Store in session state
            st.session_state.logs = logs
            st.session_state.mc_results = mc_results
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Well Log Visualization", 
                "Crossplots", 
                "AVO Analysis", 
                "Rock Physics Templates",
                "Wedge Modeling"
            ])
            
            with tab1:
                # Well log visualization code...
                pass
                
            with tab2:
                # Crossplots code...
                pass
                
            with tab3:
                # AVO Analysis code...
                pass
                
            with tab4:
                # Rock Physics Templates code...
                pass
                
            with tab5:
                # Wedge Modeling code...
                pass
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
