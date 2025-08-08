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
import logging
from typing import Tuple, Dict, Optional, Union, List
from pydantic import BaseModel, validator
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import rockphypy with error handling
try:
    from rockphypy import QI, GM, Fluid
    rockphypy_available = True
except ImportError:
    rockphypy_available = False
    logger.warning("rockphypy package not available - RPT functionality will be limited")

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced Rock Physics & AVO Modeling")

# Title and description
st.title("Enhanced Rock Physics & AVO Modeling Tool")
st.markdown("""
This app performs advanced rock physics modeling and AVO analysis with multiple models, 
visualization options, uncertainty analysis, sonic log prediction, and seismic inversion feasibility assessment.
""")

# Available colormaps for seismic displays
seismic_colormaps = ['seismic', 'RdBu', 'bwr', 'coolwarm', 'viridis', 'plasma']

# ==============================================
# Data Models for Parameter Validation
# ==============================================

class FluidProperties(BaseModel):
    rho: float
    k: float
    rho_std: float = 0.0
    k_std: float = 0.0

    @validator('rho', 'k', 'rho_std', 'k_std')
    def must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('must be non-negative')
        return v

class MineralProperties(BaseModel):
    rho: float
    k: float
    mu: float

    @validator('rho', 'k', 'mu')
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('must be positive')
        return v

class ModelParameters(BaseModel):
    phi_c: Optional[float] = None
    Cn: Optional[int] = None
    P: Optional[float] = None
    c: Optional[float] = None
    lithology: Optional[str] = None

# ==============================================
# Enhanced Rock Physics Model Functions
# ==============================================

def validate_rock_properties(vp: float, vs: float, rho: float) -> None:
    """Validate basic rock physics properties."""
    if vp <= 0 or vs <= 0 or rho <= 0:
        raise ValueError("Velocities and density must be positive")
    if vp <= vs:
        raise ValueError("P-wave velocity must be greater than S-wave velocity")

@st.cache_data
def frm(vp1: float, vs1: float, rho1: float, rho_f1: float, k_f1: float, 
       rho_f2: float, k_f2: float, k0: float, mu0: float, phi: float) -> Tuple[float, float, float, float]:
    """Gassmann's Fluid Substitution"""
    validate_rock_properties(vp1, vs1, rho1)
    
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

@st.cache_data
def critical_porosity_model(vp1: float, vs1: float, rho1: float, rho_f1: float, k_f1: float, 
                          rho_f2: float, k_f2: float, k0: float, mu0: float, phi: float, 
                          phi_c: float) -> Tuple[float, float, float, float]:
    """Critical Porosity Model (Nur et al.)"""
    validate_rock_properties(vp1, vs1, rho1)
    
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
    mu2 = mudry  # Shear modulus not affected by fluid in Gassmann
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2
    vs2 = np.sqrt(mu2/rho2)
    
    return vp2*1000, vs2*1000, rho2, k_s2

@st.cache_data
def hertz_mindlin_model(vp1: float, vs1: float, rho1: float, rho_f1: float, k_f1: float,
                       rho_f2: float, k_f2: float, k0: float, mu0: float, phi: float,
                       Cn: int, P: float) -> Tuple[float, float, float, float]:
    """Hertz-Mindlin contact theory model"""
    validate_rock_properties(vp1, vs1, rho1)
    
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

@st.cache_data
def dvorkin_nur_model(vp1: float, vs1: float, rho1: float, rho_f1: float, k_f1: float,
                     rho_f2: float, k_f2: float, k0: float, mu0: float, phi: float,
                     Cn: int = 9, P: float = 10, phi_c: float = 0.4) -> Tuple[float, float, float, float]:
    """Dvorkin-Nur Soft Sand model for unconsolidated sands"""
    validate_rock_properties(vp1, vs1, rho1)
    
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

@st.cache_data
def raymer_hunt_model(vp1: float, vs1: float, rho1: float, rho_f1: float, k_f1: float,
                     rho_f2: float, k_f2: float, k0: float, mu0: float, phi: float) -> Tuple[float, float, float, None]:
    """Raymer-Hunt-Gardner empirical model"""
    validate_rock_properties(vp1, vs1, rho1)
    
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

@st.cache_data
def xu_payne_model(vp1: float, vs1: float, rho1: float, phi: float, vsh: float,
                  c: float = 1.0, k0: float = 37.0, mu0: float = 44.0,
                  k_sh: float = 15.0, mu_sh: float = 5.0) -> Tuple[float, float, float, float]:
    """Xu-Payne laminated sand-shale model"""
    validate_rock_properties(vp1, vs1, rho1)
    
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

@st.cache_data
def greenberg_castagna(vp: float, vs: float, rho: float, phi: float, sw: float,
                      lithology: str = 'sandstone') -> Tuple[float, float, float, None]:
    """Greenberg-Castagna empirical Vp-Vs relationships"""
    validate_rock_properties(vp, vs, rho)
    
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
def ricker_wavelet(frequency: float, length: float = 0.128, dt: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Ricker wavelet"""
    t = np.linspace(-length/2, length/2, int(length/dt))
    y = (1 - 2*(np.pi**2)*(frequency**2)*(t**2)) * np.exp(-(np.pi**2)*(frequency**2)*(t**2))
    return t, y

@st.cache_data
def smith_gidlow(vp1: float, vp2: float, vs1: float, vs2: float, rho1: float, rho2: float) -> Tuple[float, float, float]:
    """Calculate Smith-Gidlow AVO attributes (intercept, gradient)"""
    # Calculate reflectivities
    rp = 0.5 * (vp2 - vp1) / (vp2 + vp1) + 0.5 * (rho2 - rho1) / (rho2 + rho1)
    rs = 0.5 * (vs2 - vs1) / (vs2 + vs1) + 0.5 * (rho2 - rho1) / (rho2 + rho1)
    
    # Smith-Gidlow coefficients
    intercept = rp
    gradient = rp - 2 * rs
    fluid_factor = rp + 1.16 * (vp1/vs1) * rs
    
    return intercept, gradient, fluid_factor

@st.cache_data
def calculate_reflection_coefficients(vp1: float, vp2: float, vs1: float, vs2: float,
                                    rho1: float, rho2: float, angle: float) -> float:
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

@st.cache_data
def fit_avo_curve(angles: np.ndarray, rc_values: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
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
# Enhanced Helper Functions
# ==============================================

@st.cache_data
def monte_carlo_iteration(params: Dict) -> Dict:
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

def parallel_monte_carlo(logs: pd.DataFrame, model_func: callable, params: Dict,
                        iterations: int = 100) -> Dict[str, List[float]]:
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

def process_monte_carlo_chunk(logs: pd.DataFrame, model_func: callable, args: List) -> Dict:
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

def create_interactive_crossplot(logs: pd.DataFrame, depth_range: Tuple[float, float] = None) -> figure:
    """Create interactive crossplot with Bokeh"""
    if depth_range:
        logs = logs[(logs['DEPTH'] >= depth_range[0]) & (logs['DEPTH'] <= depth_range[1])]
    
    # Convert numeric factors to strings
    logs['LFC_MIX'] = logs['LFC_MIX'].astype(str)  # Convert to strings
    
    source = ColumnDataSource(logs)
    palette = Category10[10]
    
    p = figure(tools="pan,wheel_zoom,box_zoom,reset,hover,save",
               title="Interactive Crossplot")
    
    # Use string factors
    factors = sorted(logs['LFC_MIX'].unique().astype(str))
    cmap = factor_cmap('LFC_MIX', palette=palette, factors=factors)
    
    p.scatter('IP_FRMMIX', 'VPVS_FRMMIX', size=8, source=source,
              color=cmap, alpha=0.6, legend_field='LFC_MIX')
    return p

def create_interactive_3d_crossplot(logs: pd.DataFrame, x_col: str = 'IP', y_col: str = 'VPVS',
                                   z_col: str = 'RHO', color_col: str = 'LFC_B') -> go.Figure:
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
    
    # Update layout with valid scene properties
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

def predict_sonic_logs(logs: pd.DataFrame, features: List[str], target_vp: str = 'VP',
                      target_vs: str = 'VS', model_type: str = 'Random Forest') -> Optional[Dict]:
    """Predict VP and VS using selected machine learning model"""
    try:
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
    except Exception as e:
        logger.error(f"Sonic prediction failed: {str(e)}")
        st.error(f"Sonic prediction failed: {str(e)}")
        return None

def seismic_inversion_feasibility(logs: pd.DataFrame, wavelet_freq: float) -> Optional[Dict]:
    """Analyze seismic inversion feasibility"""
    try:
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
    except Exception as e:
        logger.error(f"Inversion feasibility analysis failed: {str(e)}")
        st.error(f"Inversion feasibility analysis failed: {str(e)}")
        return None

def get_table_download_link(df: pd.DataFrame, filename: str = "results.csv") -> str:
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def plot_rpt_with_gassmann(title: str, fluid: str = 'gas', 
                         rho_qz: float = 2.65, k_qz: float = 37.0, mu_qz: float = 44.0,
                         rho_sh: float = 2.81, k_sh: float = 15.0, mu_sh: float = 5.0,
                         rho_b: float = 1.09, k_b: float = 2.8,
                         rho_o: float = 0.78, k_o: float = 0.94,
                         rho_g: float = 0.25, k_g: float = 0.06,
                         phi_c: float = 0.4, Cn: int = 9, sigma: float = 20, f: float = 0.5,
                         sw: float = 0.8, so: float = 0.15, sg: float = 0.05,
                         sand_cutoff: float = 0.12,
                         depth_range: Tuple[float, float] = None,
                         logs: pd.DataFrame = None) -> None:
    """Enhanced RPT plotting with data points from logs."""
    try:
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
        
        # 4. Plot actual data points from logs if available
        if logs is not None and all(col in logs.columns for col in ['VP_FRMMIX', 'VS_FRMMIX', 'RHO_FRMMIX', 'LFC_MIX']):
            # Filter logs by depth range if specified
            if depth_range:
                logs = logs[(logs['DEPTH'] >= depth_range[0]) & (logs['DEPTH'] <= depth_range[1])]
            
            # Calculate P-Impedance and Vp/Vs for the mixed case
            ip_mix = logs['VP_FRMMIX'] * logs['RHO_FRMMIX']
            vpvs_mix = logs['VP_FRMMIX'] / logs['VS_FRMMIX']
            lfc = logs['LFC_MIX']
            
            # Convert to appropriate units for RPT (km/s*g/cc)
            ip_mix_km = ip_mix / 1000  # Convert to km/s*g/cc
            vpvs_mix_km = vpvs_mix  # No conversion needed
            
            # Plot the points with facies coloring
            scatter = plt.scatter(ip_mix_km, vpvs_mix_km, c=lfc, 
                                cmap=cmap_facies, s=50, 
                                edgecolors='k', alpha=0.7,
                                vmin=0, vmax=5)
            
            # Add colorbar for facies
            cbar = plt.colorbar(scatter)
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
        
    except Exception as e:
        logger.error(f"Error generating RPT plot: {str(e)}")
        st.error(f"Error generating RPT plot: {str(e)}")

# ==============================================
# Main Application with Enhanced Features
# ==============================================

def main():
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
        
        # Mineral properties
        with st.expander("Mineral Properties"):
            col1, col2 = st.columns(2)
            with col1:
                rho_qz = st.number_input("Quartz Density (g/cc)", value=2.65, step=0.01, key="rho_qz")
                k_qz = st.number_input("Quartz Bulk Modulus (GPa)", value=37.0, step=0.1, key="k_qz")
                mu_qz = st.number_input("Quartz Shear Modulus (GPa)", value=44.0, step=0.1, key="mu_qz")
            with col2:
                rho_sh = st.number_input("Shale Density (g/cc)", value=2.81, step=0.01, key="rho_sh")
                k_sh = st.number_input("Shale Bulk Modulus (GPa)", value=15.0, step=0.1, key="k_sh")
                mu_sh = st.number_input("Shale Shear Modulus (GPa)", value=5.0, step=0.1, key="mu_sh")
        
        # Additional parameters for selected models
        if model_choice == "Critical Porosity Model (Nur)":
            critical_porosity = st.slider("Critical Porosity (φc)", 0.3, 0.5, 0.4, 0.01, key="phi_c")
        elif model_choice in ["Contact Theory (Hertz-Mindlin)", "Dvorkin-Nur Soft Sand Model"]:
            coordination_number = st.slider("Coordination Number", 6, 12, 9, key="Cn")
            effective_pressure = st.slider("Effective Pressure (MPa)", 1, 50, 10, key="P")
            if model_choice == "Dvorkin-Nur Soft Sand Model":
                critical_porosity = st.slider("Critical Porosity (φc)", 0.3, 0.5, 0.4, 0.01, key="phi_c_sand")
        elif model_choice == "Xu-Payne Laminated Model":
            lamination_factor = st.slider("Lamination Factor (c)", 0.1, 2.0, 1.0, 0.1, key="c")
        elif model_choice == "Greenberg-Castagna Empirical":
            lithology_type = st.selectbox("Lithology Type", ['sandstone', 'shale', 'carbonate', 'dolomite'], key="lithology")
        
        # Rockphypy specific parameters
        if model_choice in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
            st.subheader("RPT Model Parameters")
            rpt_phi_c = st.slider("RPT Critical Porosity", 0.3, 0.5, 0.4, 0.01, key="rpt_phi_c")
            rpt_Cn = st.slider("RPT Coordination Number", 6.0, 12.0, 8.6, 0.1, key="rpt_Cn")
            rpt_sigma = st.slider("RPT Effective Stress (MPa)", 1, 50, 20, key="rpt_sigma")
        
        # Fluid properties with uncertainty ranges
        with st.expander("Fluid Properties"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Brine**")
                rho_b = st.number_input("Brine Density (g/cc)", value=1.09, step=0.01, key="rho_b")
                k_b = st.number_input("Brine Bulk Modulus (GPa)", value=2.8, step=0.1, key="k_b")
                rho_b_std = st.number_input("Brine Density Std Dev", value=0.05, step=0.01, min_value=0.0, key="rho_b_std")
                k_b_std = st.number_input("Brine Bulk Modulus Std Dev", value=0.1, step=0.01, min_value=0.0, key="k_b_std")
            with col2:
                st.markdown("**Oil**")
                rho_o = st.number_input("Oil Density (g/cc)", value=0.78, step=0.01, key="rho_o")
                k_o = st.number_input("Oil Bulk Modulus (GPa)", value=0.94, step=0.1, key="k_o")
                rho_o_std = st.number_input("Oil Density Std Dev", value=0.05, step=0.01, min_value=0.0, key="rho_o_std")
                k_o_std = st.number_input("Oil Bulk Modulus Std Dev", value=0.05, step=0.01, min_value=0.0, key="k_o_std")
            with col3:
                st.markdown("**Gas**")
                rho_g = st.number_input("Gas Density (g/cc)", value=0.25, step=0.01, key="rho_g")
                k_g = st.number_input("Gas Bulk Modulus (GPa)", value=0.06, step=0.01, key="k_g")
                rho_g_std = st.number_input("Gas Density Std Dev", value=0.02, step=0.01, min_value=0.0, key="rho_g_std")
                k_g_std = st.number_input("Gas Bulk Modulus Std Dev", value=0.01, step=0.01, min_value=0.0, key="k_g_std")
        
        # Saturation controls
        with st.expander("Saturation Settings"):
            sw_default = 0.8
            so_default = 0.15
            sg_default = 0.05
            
            sw = st.slider("Water Saturation (Sw)", 0.0, 1.0, sw_default, 0.01, key="sw")
            remaining = max(0.0, 1.0 - sw)  # Ensure remaining is never negative
            so = st.slider(
                "Oil Saturation (So)", 
                0.0, 
                remaining, 
                min(so_default, remaining) if remaining > 0 else 0.0, 
                0.01,
                key="so"
            )
            sg = remaining - so
            
            # Display actual saturations (in case adjustments were made)
            st.write(f"Current saturations: Sw={sw:.2f}, So={so:.2f}, Sg={sg:.2f}")
        
        # AVO modeling parameters
        with st.expander("AVO Modeling Parameters"):
            min_angle = st.slider("Minimum Angle (deg)", 0, 10, 0, key="avo_min_angle")
            max_angle = st.slider("Maximum Angle (deg)", 30, 50, 45, key="avo_max_angle")
            angle_step = st.slider("Angle Step (deg)", 1, 5, 1, key="angle_step")
            wavelet_freq = st.slider("Wavelet Frequency (Hz)", 20, 80, 50, key="wavelet_freq")
            sand_cutoff = st.slider("Sand Cutoff (VSH)", 0.0, 0.3, 0.12, step=0.01, key="sand_cutoff")
        
        # Time-Frequency Analysis Parameters
        with st.expander("Time-Frequency Analysis"):
            cwt_scales = st.slider("CWT Scales Range", 1, 100, (1, 50), key="cwt_scales")
            cwt_wavelet = st.selectbox("CWT Wavelet", ['morl', 'cmor', 'gaus', 'mexh'], index=0, key="cwt_wavelet")
        
        # Monte Carlo parameters
        with st.expander("Uncertainty Analysis"):
            mc_iterations = st.slider("Monte Carlo Iterations", 10, 1000, 100, key="mc_iterations")
            parallel_processing = st.checkbox("Use parallel processing", value=True, key="parallel_processing")
            include_uncertainty = st.checkbox("Include Uncertainty Analysis", value=False, key="include_uncertainty")
        
        # Visualization options
        with st.expander("Visualization Options"):
            selected_cmap = st.selectbox("Color Map", seismic_colormaps, index=0, key="colormap")
            show_3d_crossplot = st.checkbox("Show 3D Crossplot", value=False, key="show_3d_crossplot")
            show_interactive_3d = st.checkbox("Show Interactive 3D Plot", value=True, key="show_interactive_3d")
            show_histograms = st.checkbox("Show Histograms", value=True, key="show_histograms")
            show_smith_gidlow = st.checkbox("Show Smith-Gidlow AVO Attributes", value=True, key="show_smith_gidlow")
        
        # Advanced Modules
        with st.expander("Advanced Modules"):
            predict_sonic = st.checkbox("Enable Sonic Log Prediction", value=False, key="predict_sonic")
            if predict_sonic:
                ml_model = st.selectbox("ML Model", 
                                      ["Random Forest", "XGBoost", "Gaussian Process", "Neural Network"],
                                      index=0,
                                      key="ml_model")
            inversion_feasibility = st.checkbox("Enable Seismic Inversion Feasibility", value=False, key="inversion_feasibility")
        
        # File upload
        with st.expander("Data Input"):
            uploaded_file = st.file_uploader("Upload CSV or LAS file", type=["csv", "las"], key="file_uploader")
            if uploaded_file:
                st.success("File uploaded successfully!")
        
        # Depth range for RPT plots
        with st.expander("RPT Settings"):
            rpt_depth_range = st.slider(
                "Depth Range for RPT Plots (m)",
                0.0,
                5000.0,
                (1000.0, 3000.0),
                key="rpt_depth_range"
            )

    # Main processing
    if uploaded_file is not None:
        try:
            # Store the original file object
            original_file = uploaded_file
            
            # Process data with selected model and saturations
            logs, mc_results = process_data(
                original_file, 
                model_choice,
                include_uncertainty=include_uncertainty,
                mc_iterations=mc_iterations,
                rho_qz=rho_qz, k_qz=k_qz, mu_qz=mu_qz,
                rho_sh=rho_sh, k_sh=k_sh, mu_sh=mu_sh,
                rho_b=rho_b, k_b=k_b,
                rho_o=rho_o, k_o=k_o,
                rho_g=rho_g, k_g=k_g,
                sand_cutoff=sand_cutoff,
                sw=sw, so=so, sg=sg,  # Pass saturation values
                critical_porosity=critical_porosity if 'critical_porosity' in locals() else None,
                coordination_number=coordination_number if 'coordination_number' in locals() else None,
                effective_pressure=effective_pressure if 'effective_pressure' in locals() else None,
                lamination_factor=lamination_factor if 'lamination_factor' in locals() else None,
                lithology_type=lithology_type if 'lithology_type' in locals() else None
            )
            
            if logs is None:
                st.error("Data processing failed - check your input data")
                st.stop()
            
            # Depth range selection
            ztop, zbot = st.slider(
                "Select Depth Range", 
                float(logs.DEPTH.min()), 
                float(logs.DEPTH.max()), 
                (float(logs.DEPTH.min()), float(logs.DEPTH.max())),
                key="depth_range"
            )
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs([
                "Well Log Visualization", 
                "Crossplots", 
                "AVO Analysis", 
                "Rock Physics Templates"
            ])
            
            # Visualization
            ccc = ['#B3B3B3','blue','green','red','magenta','#996633']  # Added magenta for mixed case
            cmap_facies = colors.ListedColormap(ccc[0:len(ccc)], 'indexed')

            # Create a filtered dataframe for the selected depth range
            ll = logs.loc[(logs.DEPTH>=ztop) & (logs.DEPTH<=zbot)]
            cluster = np.repeat(np.expand_dims(ll['LFC_B'].values,1), 100, 1)

            with tab1:
                # [Previous tab1 content...]

            with tab2:
                # [Previous tab2 content...]

            with tab3:
                # [Previous tab3 content...]

            with tab4:
                if model_choice in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"] and rockphypy_available:
                    st.header("Rock Physics Templates (RPT) with Data Points")
                    
                    # Get the depth range from the slider
                    rpt_depth_range = st.session_state.get('rpt_depth_range', (1000.0, 3000.0))
                    
                    # Gas Case
                    st.subheader("Gas Case RPT")
                    plot_rpt_with_gassmann(
                        model_choice.split(' ')[0],
                        fluid='gas',
                        rho_qz=rho_qz,
                        k_qz=k_qz,
                        mu_qz=mu_qz,
                        rho_sh=rho_sh,
                        k_sh=k_sh,
                        mu_sh=mu_sh,
                        rho_b=rho_b,
                        k_b=k_b,
                        rho_o=rho_o,
                        k_o=k_o,
                        rho_g=rho_g,
                        k_g=k_g,
                        phi_c=rpt_phi_c,
                        Cn=rpt_Cn,
                        sigma=rpt_sigma,
                        sw=sw,
                        so=so,
                        sg=sg,
                        sand_cutoff=sand_cutoff,
                        depth_range=rpt_depth_range,
                        logs=logs
                    )
                    
                    # Oil Case
                    st.subheader("Oil Case RPT")
                    plot_rpt_with_gassmann(
                        model_choice.split(' ')[0],
                        fluid='oil',
                        rho_qz=rho_qz,
                        k_qz=k_qz,
                        mu_qz=mu_qz,
                        rho_sh=rho_sh,
                        k_sh=k_sh,
                        mu_sh=mu_sh,
                        rho_b=rho_b,
                        k_b=k_b,
                        rho_o=rho_o,
                        k_o=k_o,
                        rho_g=rho_g,
                        k_g=k_g,
                        phi_c=rpt_phi_c,
                        Cn=rpt_Cn,
                        sigma=rpt_sigma,
                        sw=sw,
                        so=so,
                        sg=sg,
                        sand_cutoff=sand_cutoff,
                        depth_range=rpt_depth_range,
                        logs=logs
                    )
                    
                    # Mixed Case
                    st.subheader("Mixed Case RPT")
                    plot_rpt_with_gassmann(
                        model_choice.split(' ')[0],
                        fluid='mixed',
                        rho_qz=rho_qz,
                        k_qz=k_qz,
                        mu_qz=mu_qz,
                        rho_sh=rho_sh,
                        k_sh=k_sh,
                        mu_sh=mu_sh,
                        rho_b=rho_b,
                        k_b=k_b,
                        rho_o=rho_o,
                        k_o=k_o,
                        rho_g=rho_g,
                        k_g=k_g,
                        phi_c=rpt_phi_c,
                        Cn=rpt_Cn,
                        sigma=rpt_sigma,
                        sw=sw,
                        so=so,
                        sg=sg,
                        sand_cutoff=sand_cutoff,
                        depth_range=rpt_depth_range,
                        logs=logs
                    )

        except Exception as e:
            logger.error(f"Error in main processing: {str(e)}")
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
