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
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2)
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

@st.cache_data
def process_data(
    uploaded_file, 
    model_choice,
    include_uncertainty=False,
    mc_iterations=100,
    rho_qz=2.65, k_qz=37.0, mu_qz=44.0,
    rho_sh=2.81, k_sh=15.0, mu_sh=5.0,
    rho_b=1.09, k_b=2.8,
    rho_o=0.78, k_o=0.94,
    rho_g=0.25, k_g=0.06,
    sand_cutoff=0.12,
    sw=0.8, so=0.15, sg=0.05,
    critical_porosity=None,
    coordination_number=None,
    effective_pressure=None,
    lamination_factor=None,
    lithology_type=None
):
    """Process uploaded CSV file and apply selected rock physics model"""
    try:
        # Read CSV file with robust error handling
        try:
            logs = pd.read_csv(uploaded_file)
            
            # Auto-detect depth column if not named 'DEPTH'
            depth_cols = [col for col in logs.columns if 'depth' in col.lower() or 'dept' in col.lower()]
            if depth_cols and 'DEPTH' not in logs.columns:
                logs['DEPTH'] = logs[depth_cols[0]]
            
        except Exception as e:
            st.error(f"Failed to read CSV file: {str(e)}")
            return None, None
        
        # Standardize column names (case-insensitive)
        col_mapping = {
            'vp': 'VP', 'velocity_p': 'VP', 'p_velocity': 'VP',
            'vs': 'VS', 'velocity_s': 'VS', 's_velocity': 'VS',
            'rho': 'RHO', 'density': 'RHO',
            'phi': 'PHI', 'porosity': 'PHI',
            'vsh': 'VSH', 'shale_volume': 'VSH'
        }
        
        for old_col in logs.columns:
            lower_col = old_col.lower()
            if lower_col in col_mapping:
                logs[col_mapping[lower_col]] = logs[old_col]
        
        # Verify required columns exist
        required_columns = ['DEPTH', 'VP', 'VS', 'RHO', 'PHI', 'VSH']
        missing_cols = [col for col in required_columns if col not in logs.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.info("Detected columns: " + ", ".join(logs.columns))
            return None, None
        
        # Convert units if needed
        if logs.VP.mean() < 1000:  # Probably in km/s
            logs['VP'] = logs.VP * 1000  # Convert to m/s
        if logs.VS.mean() < 1000:
            logs['VS'] = logs.VS * 1000
        if logs.RHO.mean() > 1000:  # Probably in kg/m3
            logs['RHO'] = logs.RHO / 1000  # Convert to g/cc
        
        # Validate data ranges
        valid_ranges = {
            'VP': (1000, 8000),  # m/s
            'VS': (500, 5000),    # m/s
            'RHO': (1.5, 3.0),    # g/cc
            'PHI': (0, 0.4),      # fraction
            'VSH': (0, 1)         # fraction
        }
        
        for col, (min_val, max_val) in valid_ranges.items():
            if (logs[col] < min_val).any() or (logs[col] > max_val).any():
                st.warning(f"Column {col} contains values outside typical range ({min_val}-{max_val})")
        
        # Initialize results
        mc_results = None
        
        # Calculate mixed fluid properties
        rho_mix = sw*rho_b + so*rho_o + sg*rho_g
        k_mix = 1/(sw/k_b + so/k_o + sg/k_g) if (sw+so+sg) > 0 else 0
        
        # Define fluid cases
        cases = {
            'FRMB': {'rho_f': rho_b, 'k_f': k_b},
            'FRMO': {'rho_f': rho_o, 'k_f': k_o},
            'FRMG': {'rho_f': rho_g, 'k_f': k_g},
            'FRMMIX': {'rho_f': rho_mix, 'k_f': k_mix}
        }
        
        # Process each fluid case
        for case, params in cases.items():
            vp_col = f'VP_{case}'
            vs_col = f'VS_{case}'
            rho_col = f'RHO_{case}'
            
            # Initialize result arrays
            vp_results = np.zeros(len(logs))
            vs_results = np.zeros(len(logs))
            rho_results = np.zeros(len(logs))
            
            # Process each row individually
            for i in range(len(logs)):
                try:
                    if model_choice == "Gassmann's Fluid Substitution":
                        vp, vs, rho, _ = frm(
                            logs.VP.iloc[i], logs.VS.iloc[i], logs.RHO.iloc[i],
                            1.0, 2.25,  # Original fluid properties
                            params['rho_f'], params['k_f'],
                            k_qz, mu_qz, logs.PHI.iloc[i]
                        )
                    elif model_choice == "Critical Porosity Model (Nur)":
                        vp, vs, rho, _ = critical_porosity_model(
                            logs.VP.iloc[i], logs.VS.iloc[i], logs.RHO.iloc[i],
                            1.0, 2.25,
                            params['rho_f'], params['k_f'],
                            k_qz, mu_qz, logs.PHI.iloc[i], critical_porosity
                        )
                    elif model_choice == "Contact Theory (Hertz-Mindlin)":
                        vp, vs, rho, _ = hertz_mindlin_model(
                            logs.VP.iloc[i], logs.VS.iloc[i], logs.RHO.iloc[i],
                            1.0, 2.25,
                            params['rho_f'], params['k_f'],
                            k_qz, mu_qz, logs.PHI.iloc[i],
                            coordination_number, effective_pressure
                        )
                    elif model_choice == "Dvorkin-Nur Soft Sand Model":
                        vp, vs, rho, _ = dvorkin_nur_model(
                            logs.VP.iloc[i], logs.VS.iloc[i], logs.RHO.iloc[i],
                            1.0, 2.25,
                            params['rho_f'], params['k_f'],
                            k_qz, mu_qz, logs.PHI.iloc[i],
                            coordination_number, effective_pressure, critical_porosity
                        )
                    elif model_choice == "Raymer-Hunt-Gardner Model":
                        vp, vs, rho, _ = raymer_hunt_model(
                            logs.VP.iloc[i], logs.VS.iloc[i], logs.RHO.iloc[i],
                            1.0, 2.25,
                            params['rho_f'], params['k_f'],
                            k_qz, mu_qz, logs.PHI.iloc[i]
                        )
                    elif model_choice == "Xu-Payne Laminated Model":
                        vp, vs, rho, _ = xu_payne_model(
                            logs.VP.iloc[i], logs.VS.iloc[i], logs.RHO.iloc[i],
                            logs.PHI.iloc[i], logs.VSH.iloc[i],
                            lamination_factor, k_qz, mu_qz, k_sh, mu_sh
                        )
                    elif model_choice == "Greenberg-Castagna Empirical":
                        vp, vs, rho, _ = greenberg_castagna(
                            logs.VP.iloc[i], logs.VS.iloc[i], logs.RHO.iloc[i],
                            logs.PHI.iloc[i], sw,
                            lithology_type
                        )
                    else:
                        st.error(f"Unknown model choice: {model_choice}")
                        return None, None
                    
                    vp_results[i] = vp
                    vs_results[i] = vs
                    rho_results[i] = rho
                    
                except Exception as e:
                    st.warning(f"Error processing sample {i} for {case}: {str(e)}")
                    vp_results[i] = np.nan
                    vs_results[i] = np.nan
                    rho_results[i] = np.nan
            
            # Store results
            logs[vp_col] = vp_results
            logs[vs_col] = vs_results
            logs[rho_col] = rho_results
            
            # Calculate derived properties
            logs[f'IP_{case}'] = logs[vp_col] * logs[rho_col]
            logs[f'VPVS_{case}'] = logs[vp_col] / logs[vs_col]
            
            # Create litho-fluid classes
            lfc_value = {'B': 1, 'O': 2, 'G': 3, 'MIX': 4}.get(case[-1], 0)
            logs[f'LFC_{case[-1]}'] = np.where(
                logs.VSH < sand_cutoff,
                lfc_value,
                5  # Shale
            )
        
        # Calculate original properties
        logs['IP'] = logs.VP * logs.RHO
        logs['VPVS'] = logs.VP / logs.VS
        
        # Clean up any infinite values
        logs.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return logs, mc_results
        
    except Exception as e:
        st.error(f"Error processing CSV data: {str(e)}")
        logger.error(f"CSV processing failed: {str(e)}")
        return None, None

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
                # Only show well log visualization for non-RPT models
                if model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                    # Create the well log figure
                    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))  # Added column for mixed case
                    ax[0].plot(ll.VSH, ll.DEPTH, '-g', label='Vsh')
                    ax[0].plot(ll.SW, ll.DEPTH, '-b', label='Sw')
                    ax[0].plot(ll.PHI, ll.DEPTH, '-k', label='phi')
                    ax[1].plot(ll.IP_FRMG, ll.DEPTH, '-r', label='Gas')
                    ax[1].plot(ll.IP_FRMB, ll.DEPTH, '-b', label='Brine')
                    ax[1].plot(ll.IP_FRMMIX, ll.DEPTH, '-m', label='Mixed')
                    ax[1].plot(ll.IP, ll.DEPTH, '-', color='0.5', label='Original')
                    ax[2].plot(ll.VPVS_FRMG, ll.DEPTH, '-r', label='Gas')
                    ax[2].plot(ll.VPVS_FRMB, ll.DEPTH, '-b', label='Brine')
                    ax[2].plot(ll.VPVS_FRMMIX, ll.DEPTH, '-m', label='Mixed')
                    ax[2].plot(ll.VPVS, ll.DEPTH, '-', color='0.5', label='Original')
                    im = ax[3].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=5)

                    cbar = plt.colorbar(im, ax=ax[3])
                    cbar.set_label((12*' ').join(['undef', 'brine', 'oil', 'gas', 'mixed', 'shale']))
                    cbar.set_ticks(range(0,6))
                    cbar.set_ticklabels(['']*6)

                    for i in ax[:-1]:
                        i.set_ylim(ztop,zbot)
                        i.invert_yaxis()
                        i.grid()
                        i.locator_params(axis='x', nbins=4)
                    ax[0].legend(fontsize='small', loc='lower right')
                    ax[1].legend(fontsize='small', loc='lower right')
                    ax[2].legend(fontsize='small', loc='lower right')
                    ax[0].set_xlabel("Vcl/phi/Sw"); ax[0].set_xlim(-.1,1.1)
                    ax[1].set_xlabel("Ip [m/s*g/cc]"); ax[1].set_xlim(6000,15000)
                    ax[2].set_xlabel("Vp/Vs"); ax[2].set_xlim(1.5,2)
                    ax[3].set_xlabel('LFC')
                    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([]); ax[3].set_xticklabels([])
                    
                    # Display the well log plot
                    st.pyplot(fig)

                # Sonic Log Prediction Module
                if predict_sonic and 'PHI' in logs.columns and 'VSH' in logs.columns:
                    st.header("Sonic Log Prediction")
                    
                    default_features = ['PHI', 'VSH', 'RHO']
                    if 'GR' in logs.columns:
                        default_features.append('GR')
                    if 'RT' in logs.columns:
                        default_features.append('RT')
                        
                    features = st.multiselect(
                        "Select features for prediction",
                        logs.columns.tolist(),
                        default=default_features,
                        key="sonic_features"
                    )
                    
                    if st.button("Train Prediction Models", key="train_models") and features:
                        with st.spinner("Training VP/VS prediction models..."):
                            prediction_results = predict_sonic_logs(logs, features, model_type=ml_model)
                            
                            if prediction_results:
                                st.success("Models trained successfully!")
                                col1, col2 = st.columns(2)
                                col1.metric("VP Prediction R²", f"{prediction_results['vp_r2']:.3f}")
                                col2.metric("VS Prediction R²", f"{prediction_results['vs_r2']:.3f}")
                                col1.metric("VP RMSE", f"{prediction_results['vp_rmse']:.1f} m/s")
                                col2.metric("VS RMSE", f"{prediction_results['vs_rmse']:.1f} m/s")
                                
                                # Show feature importance if available
                                if prediction_results['feature_importance'] is not None:
                                    st.subheader("Feature Importance")
                                    fig_imp, ax_imp = plt.subplots(1, 2, figsize=(12, 5))
                                    ax_imp[0].barh(prediction_results['feature_importance']['Feature'],
                                                  prediction_results['feature_importance']['VP_Importance'])
                                    ax_imp[0].set_title('VP Feature Importance')
                                    ax_imp[1].barh(prediction_results['feature_importance']['Feature'],
                                                  prediction_results['feature_importance']['VS_Importance'])
                                    ax_imp[1].set_title('VS Feature Importance')
                                    plt.tight_layout()
                                    st.pyplot(fig_imp)
                                
                                # Apply predictions to missing intervals
                                if st.checkbox("Apply predictions to missing intervals", key="apply_predictions"):
                                    missing_vp = logs.VP.isna()
                                    missing_vs = logs.VS.isna()
                                    
                                    if missing_vp.any() or missing_vs.any():
                                        X = logs[features]
                                        valid_idx = X.dropna().index
                                        X_scaled = prediction_results['scaler'].transform(X.loc[valid_idx])
                                        
                                        if missing_vp.any():
                                            logs.loc[valid_idx, 'VP_pred'] = prediction_results['vp_model'].predict(X_scaled)
                                        if missing_vs.any():
                                            logs.loc[valid_idx, 'VS_pred'] = prediction_results['vs_model'].predict(X_scaled)
                                        
                                        st.success(f"Predicted {missing_vp.sum()} VP and {missing_vs.sum()} VS values")
                                        
                                        # Plot comparison
                                        fig_pred, ax = plt.subplots(1, 2, figsize=(15, 5))
                                        if 'VP_pred' in logs.columns:
                                            ax[0].plot(logs.VP, logs.DEPTH, 'k-', label='Original VP')
                                            ax[0].plot(logs.VP_pred, logs.DEPTH, 'r--', label='Predicted VP')
                                            ax[0].set_title("VP Comparison")
                                            ax[0].legend()
                                        if 'VS_pred' in logs.columns:
                                            ax[1].plot(logs.VS, logs.DEPTH, 'k-', label='Original VS')
                                            ax[1].plot(logs.VS_pred, logs.DEPTH, 'b--', label='Predicted VS')
                                            ax[1].set_title("VS Comparison")
                                            ax[1].legend()
                                        plt.tight_layout()
                                        st.pyplot(fig_pred)

                # Seismic Inversion Feasibility Module
                if inversion_feasibility:
                    st.header("Seismic Inversion Feasibility Analysis")
                    
                    with st.spinner("Analyzing inversion feasibility..."):
                        feasibility = seismic_inversion_feasibility(logs, wavelet_freq)
                        
                        if feasibility:
                            col1, col2 = st.columns(2)
                            col1.subheader("Bandwidth Analysis")
                            col1.write(f"Nyquist Frequency: {feasibility['bandwidth']['nyquist']:.1f} Hz")
                            col1.write(f"Wavelet Bandwidth: {feasibility['bandwidth']['wavelet']:.1f} Hz")
                            col1.success("Bandwidth OK") if feasibility['bandwidth']['feasible'] else col1.error("Bandwidth too high!")
                            
                            col2.subheader("Vertical Resolution")
                            col2.write(f"Tuning Thickness: {feasibility['tuning_thickness']:.1f} m")
                            col2.write(f"Minimum Resolvable Layer: {feasibility['resolution']['vertical']:.1f} m")
                            
                            st.subheader("Property Correlation Matrix")
                            fig_corr, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(feasibility['correlation_matrix'], annot=True, cmap='coolwarm', 
                                        vmin=-1, vmax=1, ax=ax)
                            st.pyplot(fig_corr)
                            
                            st.subheader("Sensitivity Analysis")
                            sens_df = pd.DataFrame.from_dict(feasibility['sensitivity'], 
                                                            orient='index', columns=['Correlation'])
                            st.dataframe(sens_df.style.format("{:.2f}").background_gradient(cmap='coolwarm', 
                                                                                           vmin=-1, vmax=1))
                            
                            # Feasibility score (simple heuristic)
                            score = 0
                            score += 1 if feasibility['bandwidth']['feasible'] else -1
                            score += sum(np.abs(list(feasibility['sensitivity'].values())))
                            score = max(0, min(10, score * 2))  # Scale to 0-10
                            
                            st.progress(score/10)
                            st.write(f"Inversion Feasibility Score: {score:.1f}/10")
                            
                            if score < 5:
                                st.warning("Marginal feasibility - consider:")
                                st.markdown("""
                                - Higher frequency wavelet
                                - Additional conditioning (low-frequency model)
                                - Rock physics constraints
                                """)
                            else:
                                st.success("Good inversion feasibility")

            with tab2:
                # Interactive Crossplot with depth filtering
                st.header("Interactive Crossplots with Selection")
                crossplot = create_interactive_crossplot(logs, depth_range=(ztop, zbot))
                if crossplot:
                    event_result = streamlit_bokeh_events(
                        bokeh_plot=crossplot,
                        events="SELECTION_CHANGED",
                        key="crossplot",
                        refresh_on_update=False,
                        debounce_time=0,
                        override_height=500
                    )
                else:
                    st.warning("Could not generate interactive crossplot due to data issues")
                
                # Interactive 3D Crossplot
                if show_interactive_3d:
                    st.header("Interactive 3D Crossplot")
                    fig_3d = create_interactive_3d_crossplot(
                        logs[(logs['DEPTH'] >= ztop) & (logs['DEPTH'] <= zbot)],
                        x_col='IP_FRMMIX',
                        y_col='VPVS_FRMMIX',
                        z_col='RHO_FRMMIX',
                        color_col='LFC_MIX'
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                # Original 2D crossplots (now including mixed case) with depth filtering
                st.header("2D Crossplots")
                fig2, ax2 = plt.subplots(1, 5, figsize=(25, 4))  # Added column for mixed case
                
                # Filter data based on depth range
                filtered_logs = logs[(logs['DEPTH'] >= ztop) & (logs['DEPTH'] <= zbot)]
                
                ax2[0].scatter(filtered_logs.IP, filtered_logs.VPVS, 20, filtered_logs.LFC_B, marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=5)
                ax2[0].set_xlabel('IP (m/s*g/cc)')
                ax2[0].set_ylabel('Vp/Vs(unitless)')
                ax2[1].scatter(filtered_logs.IP_FRMB, filtered_logs.VPVS_FRMB, 20, filtered_logs.LFC_B, marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=5)
                ax2[1].set_xlabel('IP (m/s*g/cc)')
                ax2[1].set_ylabel('Vp/Vs(unitless)')
                ax2[2].scatter(filtered_logs.IP_FRMO, filtered_logs.VPVS_FRMO, 20, filtered_logs.LFC_O, marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=5)
                ax2[2].set_xlabel('IP (m/s*g/cc)')
                ax2[2].set_ylabel('Vp/Vs(unitless)')
                ax2[3].scatter(filtered_logs.IP_FRMG, filtered_logs.VPVS_FRMG, 20, filtered_logs.LFC_G, marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=5)
                ax2[3].set_xlabel('IP (m/s*g/cc)')
                ax2[3].set_ylabel('Vp/Vs(unitless)')
                ax2[4].scatter(filtered_logs.IP_FRMMIX, filtered_logs.VPVS_FRMMIX, 20, filtered_logs.LFC_MIX, marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=5)
                ax2[4].set_xlabel('IP (m/s*g/cc)')
                ax2[4].set_ylabel('Vp/Vs(unitless)')
                ax2[0].set_xlim(3000,16000); ax2[0].set_ylim(1.5,3)
                ax2[0].set_title('Original Data')
                ax2[1].set_title('FRM to Brine')
                ax2[2].set_title('FRM to Oil')
                ax2[3].set_title('FRM to Gas')
                ax2[4].set_title(f'FRM to Mixed (Sw={sw:.2f}, So={so:.2f}, Sg={sg:.2f})')
                st.pyplot(fig2)

                # 3D Crossplot if enabled
                if show_3d_crossplot and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                    st.header("3D Crossplot")
                    fig3d = plt.figure(figsize=(10, 8))
                    ax3d = fig3d.add_subplot(111, projection='3d')
                    
                    # Define colors for each case
                    case_colors = {
                        'B': 'blue',
                        'O': 'green',
                        'G': 'red',
                        'MIX': 'magenta'
                    }
                    
                    for case in ['B', 'O', 'G', 'MIX']:
                        mask = filtered_logs[f'LFC_{case}'] == int(case == 'B')*1 + int(case == 'O')*2 + int(case == 'G')*3 + int(case == 'MIX')*4
                        ax3d.scatter(
                            filtered_logs.loc[mask, f'IP_FRM{case}'],
                            filtered_logs.loc[mask, f'VPVS_FRM{case}'],
                            filtered_logs.loc[mask, f'RHO_FRM{case}'],
                            c=case_colors[case], label=case, alpha=0.5
                        )
                    
                    ax3d.set_xlabel('IP (m/s*g/cc)')
                    ax3d.set_ylabel('Vp/Vs')
                    ax3d.set_zlabel('Density (g/cc)')
                    ax3d.set_title('3D Rock Physics Crossplot')
                    ax3d.legend()
                    st.pyplot(fig3d)

                # Histograms if enabled
                if show_histograms and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                    st.header("Property Distributions")
                    fig_hist, ax_hist = plt.subplots(2, 2, figsize=(12, 8))
                    
                    # Define colors for each case
                    hist_colors = {
                        'B': 'blue',
                        'O': 'green',
                        'G': 'red',
                        'MIX': 'magenta'
                    }
                    
                    ax_hist[0,0].hist(filtered_logs.IP_FRMB, bins=30, alpha=0.5, label='Brine', color=hist_colors['B'])
                    ax_hist[0,0].hist(filtered_logs.IP_FRMO, bins=30, alpha=0.5, label='Oil', color=hist_colors['O'])
                    ax_hist[0,0].hist(filtered_logs.IP_FRMG, bins=30, alpha=0.5, label='Gas', color=hist_colors['G'])
                    ax_hist[0,0].hist(filtered_logs.IP_FRMMIX, bins=30, alpha=0.5, label='Mixed', color=hist_colors['MIX'])
                    ax_hist[0,0].set_xlabel('IP (m/s*g/cc)')
                    ax_hist[0,0].set_ylabel('Frequency')
                    ax_hist[0,0].legend()
                    
                    ax_hist[0,1].hist(filtered_logs.VPVS_FRMB, bins=30, alpha=0.5, label='Brine', color=hist_colors['B'])
                    ax_hist[0,1].hist(filtered_logs.VPVS_FRMO, bins=30, alpha=0.5, label='Oil', color=hist_colors['O'])
                    ax_hist[0,1].hist(filtered_logs.VPVS_FRMG, bins=30, alpha=0.5, label='Gas', color=hist_colors['G'])
                    ax_hist[0,1].hist(filtered_logs.VPVS_FRMMIX, bins=30, alpha=0.5, label='Mixed', color=hist_colors['MIX'])
                    ax_hist[0,1].set_xlabel('Vp/Vs')
                    ax_hist[0,1].legend()
                    
                    ax_hist[1,0].hist(filtered_logs.RHO_FRMB, bins=30, color=hist_colors['B'], alpha=0.7)
                    ax_hist[1,0].hist(filtered_logs.RHO_FRMO, bins=30, color=hist_colors['O'], alpha=0.7)
                    ax_hist[1,0].hist(filtered_logs.RHO_FRMG, bins=30, color=hist_colors['G'], alpha=0.7)
                    ax_hist[1,0].hist(filtered_logs.RHO_FRMMIX, bins=30, color=hist_colors['MIX'], alpha=0.7)
                    ax_hist[1,0].set_xlabel('Density (g/cc)')
                    ax_hist[1,0].set_ylabel('Frequency')
                    ax_hist[1,0].legend(['Brine', 'Oil', 'Gas', 'Mixed'])
                    
                    ax_hist[1,1].hist(filtered_logs.LFC_B, bins=[0,1,2,3,4,5,6], alpha=0.5, rwidth=0.8, align='left')
                    ax_hist[1,1].set_xlabel('Litho-Fluid Class')
                    ax_hist[1,1].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5])
                    ax_hist[1,1].set_xticklabels(['Undef','Brine','Oil','Gas','Mixed','Shale'])
                    
                    plt.tight_layout()
                    st.pyplot(fig_hist)

            with tab3:
                # AVO Modeling
                if model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                    st.header("AVO Modeling")
                    middle_top = ztop + (zbot - ztop) * 0.4
                    middle_bot = ztop + (zbot - ztop) * 0.6
                    
                    cases = ['Brine', 'Oil', 'Gas', 'Mixed']
                    case_data = {
                        'Brine': {'vp': 'VP_FRMB', 'vs': 'VS_FRMB', 'rho': 'RHO_FRMB', 'color': 'b'},
                        'Oil': {'vp': 'VP_FRMO', 'vs': 'VS_FRMO', 'rho': 'RHO_FRMO', 'color': 'g'},
                        'Gas': {'vp': 'VP_FRMG', 'vs': 'VS_FRMG', 'rho': 'RHO_FRMG', 'color': 'r'},
                        'Mixed': {'vp': 'VP_FRMMIX', 'vs': 'VS_FRMMIX', 'rho': 'RHO_FRMMIX', 'color': 'm'}
                    }
                    
                    wlt_time, wlt_amp = ricker_wavelet(wavelet_freq)
                    t_samp = np.arange(0, 0.5, 0.0001)
                    t_middle = 0.2
                    
                    fig3, (ax_wavelet, ax_avo) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 2]})
                    
                    ax_wavelet.plot(wlt_time, wlt_amp, color='purple', linewidth=2)
                    ax_wavelet.fill_between(wlt_time, wlt_amp, color='purple', alpha=0.3)
                    ax_wavelet.set_title(f"Wavelet ({wavelet_freq} Hz)")
                    ax_wavelet.set_xlabel("Time (s)")
                    ax_wavelet.set_ylabel("Amplitude")
                    ax_wavelet.grid(True)
                    
                    rc_min, rc_max = st.slider(
                        "Reflection Coefficient Range",
                        -0.5, 0.5, (-0.2, 0.2),
                        step=0.01,
                        key='rc_range'
                    )
                    
                    angles = np.arange(min_angle, max_angle + 1, angle_step)
                    
                    # Store AVO attributes for Smith-Gidlow analysis
                    avo_attributes = {'Case': [], 'Intercept': [], 'Gradient': [], 'Fluid_Factor': []}
                    
                    for case in cases:
                        vp_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VP'].values.mean()
                        vs_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VS'].values.mean()
                        rho_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'RHO'].values.mean()
                        
                        vp_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vp']].values.mean()
                        vs_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vs']].values.mean()
                        rho_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['rho']].values.mean()
                        
                        # Calculate reflection coefficients
                        rc = []
                        for angle in angles:
                            rc.append(calculate_reflection_coefficients(
                                vp_upper, vp_middle, vs_upper, vs_middle, rho_upper, rho_middle, angle
                            ))
                        
                        # Fit AVO curve to get intercept and gradient
                        intercept, gradient, _ = fit_avo_curve(angles, rc)
                        fluid_factor = intercept + 1.16 * (vp_upper/vs_upper) * (intercept - gradient)
                        
                        # Store attributes for Smith-Gidlow analysis
                        avo_attributes['Case'].append(case)
                        avo_attributes['Intercept'].append(intercept)
                        avo_attributes['Gradient'].append(gradient)
                        avo_attributes['Fluid_Factor'].append(fluid_factor)
                        
                        # Plot AVO curve
                        ax_avo.plot(angles, rc, f"{case_data[case]['color']}-", label=f"{case}")
                    
                    ax_avo.set_title("AVO Reflection Coefficients (Middle Interface)")
                    ax_avo.set_xlabel("Angle (degrees)")
                    ax_avo.set_ylabel("Reflection Coefficient")
                    ax_avo.set_ylim(rc_min, rc_max)
                    ax_avo.grid(True)
                    ax_avo.legend()
                    
                    st.pyplot(fig3)

                    # Smith-Gidlow AVO Analysis
                    if show_smith_gidlow:
                        st.header("Smith-Gidlow AVO Attributes")
                        
                        # Create DataFrame for AVO attributes
                        avo_df = pd.DataFrame(avo_attributes)
                        
                        # Display attributes table
                        if not avo_df.empty:
                            numeric_cols = avo_df.select_dtypes(include=[np.number]).columns
                            st.dataframe(avo_df.style.format("{:.4f}", subset=numeric_cols))
                        else:
                            st.warning("No AVO attributes calculated")
                        
                        # Plot intercept vs gradient
                        fig_sg, ax_sg = plt.subplots(figsize=(8, 6))
                        colors = {'Brine': 'blue', 'Oil': 'green', 'Gas': 'red', 'Mixed': 'magenta'}
                        
                        for idx, row in avo_df.iterrows():
                            ax_sg.scatter(row['Intercept'], row['Gradient'], 
                                         color=colors[row['Case']], s=100, label=row['Case'])
                            ax_sg.text(row['Intercept'], row['Gradient'], row['Case'], 
                                      fontsize=9, ha='right', va='bottom')
                        
                        # Add background classification
                        x = np.linspace(-0.5, 0.5, 100)
                        ax_sg.plot(x, -x, 'k--', alpha=0.3)  # Typical brine line
                        ax_sg.plot(x, -4*x, 'k--', alpha=0.3)  # Typical gas line
                        
                        ax_sg.set_xlabel('Intercept (A)')
                        ax_sg.set_ylabel('Gradient (B)')
                        ax_sg.set_title('Smith-Gidlow AVO Crossplot')
                        ax_sg.grid(True)
                        ax_sg.axhline(0, color='k', alpha=0.3)
                        ax_sg.axvline(0, color='k', alpha=0.3)
                        ax_sg.set_xlim(-0.3, 0.3)
                        ax_sg.set_ylim(-0.3, 0.3)
                        
                        st.pyplot(fig_sg)
                        
                        # Fluid Factor analysis
                        st.subheader("Fluid Factor Analysis")
                        fig_ff, ax_ff = plt.subplots(figsize=(8, 4))
                        ax_ff.bar(avo_df['Case'], avo_df['Fluid_Factor'], 
                                 color=[colors[c] for c in avo_df['Case']])
                        ax_ff.set_ylabel('Fluid Factor')
                        ax_ff.set_title('Fluid Factor by Fluid Type')
                        ax_ff.grid(True)
                        st.pyplot(fig_ff)

                    # Time-Frequency Analysis of Synthetic Gathers (Fixed)
                    st.header("Time-Frequency Analysis of Synthetic Gathers")
                    
                    # Generate synthetic gathers for time-frequency analysis
                    all_gathers = {}
                    for case in cases:
                        # Get average properties for upper layer (shale)
                        vp_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VP'].values.mean()
                        vs_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VS'].values.mean()
                        rho_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'RHO'].values.mean()
                        
                        # Get average properties for middle layer (sand with fluid substitution)
                        vp_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vp']].values.mean()
                        vs_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vs']].values.mean()
                        rho_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['rho']].values.mean()
                        
                        syn_gather = []
                        for angle in angles:
                            rc = calculate_reflection_coefficients(
                                vp_upper, vp_middle, vs_upper, vs_middle, rho_upper, rho_middle, angle
                            )
                            
                            # Create reflectivity series
                            rc_series = np.zeros(len(t_samp))
                            idx_middle = np.argmin(np.abs(t_samp - t_middle))
                            rc_series[idx_middle] = rc
                            
                            # Convolve with wavelet
                            syn_trace = np.convolve(rc_series, wlt_amp, mode='same')
                            syn_gather.append(syn_trace)
                        
                        all_gathers[case] = np.array(syn_gather)
                    
                    # Plot frequency domain analysis
                    st.subheader("Frequency Domain Analysis (FFT)")
                    fig_freq, ax_freq = plt.subplots(1, 4, figsize=(24, 5))
                    
                    for idx, case in enumerate(cases):
                        syn_gather = all_gathers[case]
                        
                        # Time range filtering
                        time_mask = (t_samp >= 0.15) & (t_samp <= 0.25)
                        t_samp_filtered = t_samp[time_mask]
                        
                        # Compute FFT parameters
                        n = syn_gather.shape[1]
                        dt = t_samp[1] - t_samp[0]
                        freqs = np.fft.rfftfreq(n, dt)
                        
                        # Initialize array to store frequency spectra (time vs frequency)
                        freq_spectra = np.zeros((len(t_samp_filtered), len(freqs)))
                        
                        # Calculate FFT for each time sample across all angles
                        for i, t in enumerate(t_samp_filtered):
                            time_idx = np.where(t_samp == t)[0][0]
                            time_slice = syn_gather[:, time_idx]
                            
                            # Apply Hanning window to reduce spectral leakage
                            window = np.hanning(len(time_slice))
                            windowed_signal = time_slice * window
                            
                            # Compute FFT and take magnitude
                            spectrum = np.abs(np.fft.rfft(windowed_signal))
                            
                            # Handle case where spectrum length doesn't match frequency bins
                            min_len = min(len(spectrum), len(freqs))
                            freq_spectra[i, :min_len] = spectrum[:min_len]
                        
                        # Normalize for better visualization
                        if np.max(freq_spectra) > 0:
                            freq_spectra = freq_spectra / np.max(freq_spectra)
                        
                        # Create frequency-based color coding
                        X, Y = np.meshgrid(freqs, t_samp_filtered)
                        
                        # Plot with frequency-based color
                        im = ax_freq[idx].pcolormesh(
                            X, Y, freq_spectra,
                            cmap='jet',
                            shading='auto',
                            vmin=0,
                            vmax=1
                        )
                        
                        ax_freq[idx].set_title(f"{case} Case Frequency Spectrum")
                        ax_freq[idx].set_xlabel("Frequency (Hz)")
                        ax_freq[idx].set_ylabel("Time (s)")
                        ax_freq[idx].set_ylim(0.25, 0.15)  # Inverted for seismic display
                        ax_freq[idx].set_xlim(0, wavelet_freq * 3)  # Focus on relevant frequencies
                        
                        plt.colorbar(im, ax=ax_freq[idx], label='Normalized Amplitude')
                    
                    plt.tight_layout()
                    st.pyplot(fig_freq)

                    # Plot CWT analysis
                    st.subheader("Time-Frequency Analysis (CWT)")
                    try:
                        scales = np.arange(cwt_scales[0], cwt_scales[1]+1)
                        # Convert scales to approximate frequencies
                        freqs = pywt.scale2frequency(cwt_wavelet, scales) / (t_samp[1]-t_samp[0])
                        
                        fig_cwt, ax_cwt = plt.subplots(3, 4, figsize=(24, 12))
                        
                        # Time range filtering
                        time_mask = (t_samp >= 0.15) & (t_samp <= 0.25)
                        t_samp_filtered = t_samp[time_mask]
                        
                        # Store CWT magnitudes at t=0.20s for each case (using raw magnitudes)
                        cwt_at_020s = {'Case': [], 'Frequency': [], 'Magnitude': []}
                        
                        for col_idx, case in enumerate(cases):
                            syn_gather = all_gathers[case]
                            
                            # Initialize array to store CWT magnitudes (time vs frequency)
                            cwt_magnitudes = np.zeros((len(t_samp_filtered), len(freqs)))
                            
                            for i, t in enumerate(t_samp_filtered):
                                time_idx = np.where(t_samp == t)[0][0]
                                trace = syn_gather[:, time_idx]  # All angles at this time sample
                                if len(trace) == 0:
                                    continue
                                    
                                coefficients, _ = pywt.cwt(trace, scales, cwt_wavelet, 
                                                         sampling_period=t_samp[1]-t_samp[0])
                                
                                if coefficients.size > 0:
                                    # Sum across angles for each scale (store raw magnitudes)
                                    cwt_magnitudes[i, :] = np.sum(np.abs(coefficients), axis=1)
                            
                            if cwt_magnitudes.size == 0:
                                st.warning(f"No valid CWT data for {case} case")
                                continue
                                
                            # Find global max for consistent normalization in display (not used in exported data)
                            global_max = np.max(cwt_magnitudes) if np.max(cwt_magnitudes) > 0 else 1
                            
                            # Create frequency-based color coding
                            X, Y = np.meshgrid(freqs, t_samp_filtered)
                            
                            # Plot CWT magnitude with frequency-based color (inverted y-axis)
                            im = ax_cwt[0, col_idx].pcolormesh(
                                X, Y, cwt_magnitudes/global_max,  # Normalized only for display
                                shading='auto',
                                cmap='jet', 
                                vmin=0, 
                                vmax=1
                            )
                            ax_cwt[0, col_idx].set_title(f"{case} - CWT Magnitude")
                            ax_cwt[0, col_idx].set_xlabel("Frequency (Hz)")
                            ax_cwt[0, col_idx].set_ylabel("Time (s)")
                            ax_cwt[0, col_idx].set_ylim(0.25, 0.15)  # Inverted y-axis
                            ax_cwt[0, col_idx].set_xlim(0, wavelet_freq * 3)
                            plt.colorbar(im, ax=ax_cwt[0, col_idx], label='Normalized Magnitude')
                            
                            # Plot time series at middle angle (inverted y-axis)
                            mid_angle_idx = len(angles) // 2
                            time_series = syn_gather[mid_angle_idx, time_mask]
                            ax_cwt[1, col_idx].plot(t_samp_filtered, time_series, 'k-')
                            ax_cwt[1, col_idx].set_title(f"{case} - Time Series (@ {angles[mid_angle_idx]}°)")
                            ax_cwt[1, col_idx].set_xlabel("Time (s)")
                            ax_cwt[1, col_idx].set_ylabel("Amplitude")
                            ax_cwt[1, col_idx].grid(True)
                            ax_cwt[1, col_idx].set_xlim(0.15, 0.25)
                            ax_cwt[1, col_idx].set_ylim(0.25, 0.15)  # Inverted y-axis
                            
                            # Plot dominant frequency (inverted y-axis)
                            if cwt_magnitudes.size > 0:
                                max_freq_indices = np.argmax(cwt_magnitudes, axis=1)
                                dominant_freqs = freqs[max_freq_indices]
                                
                                ax_cwt[2, col_idx].plot(t_samp_filtered, dominant_freqs, 'r-')
                                ax_cwt[2, col_idx].set_title(f"{case} - Dominant Frequency")
                                ax_cwt[2, col_idx].set_xlabel("Time (s)")
                                ax_cwt[2, col_idx].set_ylabel("Frequency (Hz)")
                                ax_cwt[2, col_idx].grid(True)
                                ax_cwt[2, col_idx].set_ylim(wavelet_freq * 3, 0)  # Inverted y-axis
                            
                            # Extract CWT magnitudes at t=0.20s (using raw magnitudes)
                            time_target = 0.20
                            time_idx = np.argmin(np.abs(t_samp_filtered - time_target))
                            
                            if time_idx < len(t_samp_filtered):
                                cwt_at_020s['Case'].extend([case] * len(freqs))
                                cwt_at_020s['Frequency'].extend(freqs)
                                cwt_at_020s['Magnitude'].extend(cwt_magnitudes[time_idx, :])  # Raw magnitudes
                        
                        plt.tight_layout()
                        st.pyplot(fig_cwt)              
                        # Plot Frequency vs. Magnitude at t=0.20s (using raw magnitudes)
                        if len(cwt_at_020s['Frequency']) > 0:
                            st.subheader("CWT Frequency vs. Magnitude at t=0.20s (Raw Magnitudes)")
                            fig_freq_mag, ax_freq_mag = plt.subplots(figsize=(10, 5))
                            
                            for case in cases:
                                case_mask = np.array(cwt_at_020s['Case']) == case
                                freqs_case = np.array(cwt_at_020s['Frequency'])[case_mask]
                                mags_case = np.array(cwt_at_020s['Magnitude'])[case_mask]  # Raw magnitudes
                                
                                if len(freqs_case) > 0:
                                    ax_freq_mag.plot(freqs_case, mags_case, label=case)
                            
                            ax_freq_mag.set_xlabel("Frequency (Hz)")
                            ax_freq_mag.set_ylabel("Magnitude (Raw Sum)")
                            ax_freq_mag.set_title("CWT Magnitude Spectrum at t=0.20s (Raw Values)")
                            ax_freq_mag.legend()
                            ax_freq_mag.grid(True)
                            ax_freq_mag.set_xlim(0, wavelet_freq * 3)
                            
                            st.pyplot(fig_freq_mag)
                        
                    except Exception as e:
                        st.error(f"Error in CWT analysis: {str(e)}")

                    # Plot spectral comparison
                    st.subheader("Spectral Comparison at Selected Angles")
                    selected_angles = st.multiselect(
                        "Select angles to compare spectra",
                        angles.tolist(),
                        default=[angles[0], angles[len(angles)//2], angles[-1]],
                        key='spectral_angles'
                    )
                    
                    if selected_angles:
                        # Time range filtering
                        time_mask = (t_samp >= 0.15) & (t_samp <= 0.25)
                        t_samp_filtered = t_samp[time_mask]
                        
                        fig_compare = plt.figure(figsize=(12, 8))
                        ax_compare = fig_compare.add_subplot(111, projection='3d')
                        
                        # Create colormap based on frequency
                        norm = plt.Normalize(0, wavelet_freq * 3)
                        cmap = plt.get_cmap('jet')
                        
                        for case in cases:
                            syn_gather = all_gathers[case]
                            
                            for angle in selected_angles:
                                angle_idx = np.where(angles == angle)[0][0]
                                if angle_idx >= syn_gather.shape[0]:
                                    continue
                                    
                                trace = syn_gather[angle_idx, time_mask]
                                
                                # FFT
                                spectrum = np.abs(np.fft.rfft(trace))
                                freqs = np.fft.rfftfreq(len(trace), t_samp[1]-t_samp[0])
                                
                                # Filter frequencies
                                freq_mask = (freqs >= 0) & (freqs <= wavelet_freq * 3)
                                freqs_filtered = freqs[freq_mask]
                                spectrum_filtered = spectrum[freq_mask]
                                
                                # Normalize spectrum
                                if np.max(spectrum_filtered) > 0:
                                    spectrum_filtered = spectrum_filtered / np.max(spectrum_filtered)
                                
                                # Create color array based on frequency
                                colors = cmap(norm(freqs_filtered))
                                
                                # Plot each frequency component separately with its own color
                                for i in range(len(freqs_filtered)-1):
                                    ax_compare.plot(
                                        [angle, angle],  # X (angle)
                                        [freqs_filtered[i], freqs_filtered[i+1]],  # Y (frequency)
                                        [spectrum_filtered[i], spectrum_filtered[i+1]],  # Z (amplitude)
                                        color=colors[i],
                                        linewidth=2
                                    )
                        
                        ax_compare.set_title("3D Spectral Comparison (Color by Frequency)")
                        ax_compare.set_xlabel("Angle (degrees)")
                        ax_compare.set_ylabel("Frequency (Hz)")
                        ax_compare.set_zlabel("Normalized Amplitude")
                        ax_compare.set_xlim(min(selected_angles), max(selected_angles))
                        ax_compare.set_ylim(0, wavelet_freq * 3)
                        ax_compare.set_zlim(0, 1)
                        
                        # Add colorbar
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])
                        fig_compare.colorbar(sm, ax=ax_compare, label='Frequency (Hz)', shrink=0.6)
                        
                        st.pyplot(fig_compare)

                    # Synthetic gathers
                    st.header("Synthetic Seismic Gathers (Middle Interface)")
                    time_min, time_max = st.slider(
                        "Time Range for Synthetic Gathers (s)",
                        0.0, 0.5, (0.15, 0.25),
                        step=0.01,
                        key='time_range'
                    )
                    
                    fig4, ax4 = plt.subplots(1, 4, figsize=(24, 5))  # Increased figure size
                    
                    for idx, case in enumerate(cases):
                        syn_gather = all_gathers[case]
                        
                        extent = [angles[0], angles[-1], t_samp[-1], t_samp[0]]
                        im = ax4[idx].imshow(syn_gather.T, aspect='auto', extent=extent,
                                           cmap=selected_cmap, vmin=-np.max(np.abs(syn_gather)), 
                                           vmax=np.max(np.abs(syn_gather)))
                        
                        props_text = f"Vp: {logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vp']].values.mean():.0f} m/s\n" \
                                    f"Vs: {logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vs']].values.mean():.0f} m/s\n" \
                                    f"Rho: {logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['rho']].values.mean():.2f} g/cc"
                        ax4[idx].text(0.05, 0.95, props_text, transform=ax4[idx].transAxes,
                                     fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
                        
                        ax4[idx].set_title(f"{case} Case", fontweight='bold')
                        ax4[idx].set_xlabel("Angle (degrees)")
                        ax4[idx].set_ylabel("Time (s)")
                        ax4[idx].set_ylim(time_max, time_min)
                        
                        plt.colorbar(im, ax=ax4[idx], label='Amplitude')
                    
                    plt.tight_layout()
                    st.pyplot(fig4)

            with tab4:
                # Rock Physics Templates (RPT)
                if model_choice in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"] and rockphypy_available:
                    st.header("Rock Physics Templates (RPT) with Gassmann Fluid Substitution")
                    
                    # Get the depth range from the slider
                    rpt_depth_range = st.session_state.get('rpt_depth_range', (1000.0, 3000.0))
                    
                    # Display Gas Case RPT with Gassmann points
                    st.subheader("Gas Case RPT with Gassmann Fluid Substitution")
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
                    
                    # Display Oil Case RPT with Gassmann points
                    st.subheader("Oil Case RPT with Gassmann Fluid Substitution")
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
                    
                    # Display Mixed Case RPT with Gassmann points
                    st.subheader("Mixed Case RPT with Gassmann Fluid Substitution")
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

                # Uncertainty Analysis Results
                if include_uncertainty and mc_results and model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                    st.header("Uncertainty Analysis Results")
                    
                    # Create summary statistics
                    mc_df = pd.DataFrame(mc_results)
                    summary_stats = mc_df.describe().T
                    
                    st.subheader("Monte Carlo Simulation Statistics")
                    if not summary_stats.empty:
                        numeric_cols = summary_stats.select_dtypes(include=[np.number]).columns
                        st.dataframe(summary_stats.style.format("{:.2f}", subset=numeric_cols))
                    else:
                        st.warning("No statistics available - check your Monte Carlo simulation parameters")
                    
                    # Plot uncertainty distributions
                    st.subheader("Property Uncertainty Distributions")
                    fig_unc, ax_unc = plt.subplots(2, 2, figsize=(12, 8))
                    
                    # VP distribution
                    ax_unc[0,0].hist(mc_results['VP'], bins=30, color='blue', alpha=0.7)
                    ax_unc[0,0].set_xlabel('VP (m/s)')
                    ax_unc[0,0].set_ylabel('Frequency')
                    ax_unc[0,0].set_title('P-wave Velocity Distribution')
                    
                    # VS distribution
                    ax_unc[0,1].hist(mc_results['VS'], bins=30, color='green', alpha=0.7)
                    ax_unc[0,1].set_xlabel('VS (m/s)')
                    ax_unc[0,1].set_title('S-wave Velocity Distribution')
                    
                    # IP distribution
                    ax_unc[1,0].hist(mc_results['IP'], bins=30, color='red', alpha=0.7)
                    ax_unc[1,0].set_xlabel('IP (m/s*g/cc)')
                    ax_unc[1,0].set_ylabel('Frequency')
                    ax_unc[1,0].set_title('Acoustic Impedance Distribution')
                    
                    # Vp/Vs distribution
                    ax_unc[1,1].hist(mc_results['VPVS'], bins=30, color='purple', alpha=0.7)
                    ax_unc[1,1].set_xlabel('Vp/Vs')
                    ax_unc[1,1].set_title('Vp/Vs Ratio Distribution')
                    
                    plt.tight_layout()
                    st.pyplot(fig_unc)
                    
                    # AVO attribute uncertainty
                    st.subheader("AVO Attribute Uncertainty")
                    fig_avo_unc, ax_avo_unc = plt.subplots(1, 3, figsize=(15, 4))
                    
                    # Intercept distribution
                    ax_avo_unc[0].hist(mc_results['Intercept'], bins=30, color='blue', alpha=0.7)
                    ax_avo_unc[0].set_xlabel('Intercept')
                    ax_avo_unc[0].set_ylabel('Frequency')
                    ax_avo_unc[0].set_title('Intercept Distribution')
                    
                    # Gradient distribution
                    ax_avo_unc[1].hist(mc_results['Gradient'], bins=30, color='green', alpha=0.7)
                    ax_avo_unc[1].set_xlabel('Gradient')
                    ax_avo_unc[1].set_title('Gradient Distribution')
                    
                    # Fluid Factor distribution
                    ax_avo_unc[2].hist(mc_results['Fluid_Factor'], bins=30, color='red', alpha=0.7)
                    ax_avo_unc[2].set_xlabel('Fluid Factor')
                    ax_avo_unc[2].set_title('Fluid Factor Distribution')
                    
                    plt.tight_layout()
                    st.pyplot(fig_avo_unc)
                    
                    # Crossplot of AVO attributes with uncertainty
                    st.subheader("AVO Attribute Crossplot with Uncertainty")
                    fig_avo_cross, ax_avo_cross = plt.subplots(figsize=(8, 6))
                    
                    # Plot all Monte Carlo samples
                    ax_avo_cross.scatter(mc_results['Intercept'], mc_results['Gradient'], 
                                        c=mc_results['Fluid_Factor'], cmap='coolwarm', 
                                        alpha=0.3, s=10)
                    
                    # Add colorbar
                    sc = ax_avo_cross.scatter([], [], c=[], cmap='coolwarm')
                    plt.colorbar(sc, label='Fluid Factor', ax=ax_avo_cross)
                    
                    # Add background classification
                    x = np.linspace(-0.5, 0.5, 100)
                    ax_avo_cross.plot(x, -x, 'k--', alpha=0.3)  # Typical brine line
                    ax_avo_cross.plot(x, -4*x, 'k--', alpha=0.3)  # Typical gas line
                    
                    ax_avo_cross.set_xlabel('Intercept (A)')
                    ax_avo_cross.set_ylabel('Gradient (B)')
                    ax_avo_cross.set_title('AVO Attribute Uncertainty')
                    ax_avo_cross.grid(True)
                    ax_avo_cross.axhline(0, color='k', alpha=0.3)
                    ax_avo_cross.axvline(0, color='k', alpha=0.3)
                    ax_avo_cross.set_xlim(-0.3, 0.3)
                    ax_avo_cross.set_ylim(-0.3, 0.3)
                    
                    st.pyplot(fig_avo_cross)
                    
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
