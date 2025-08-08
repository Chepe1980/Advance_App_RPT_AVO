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
import importlib.util

# Setup logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Dependency check
def check_package(package_name):
    return importlib.util.find_spec(package_name) is not None

required_packages = ['streamlit', 'numpy', 'pandas', 'matplotlib', 'scipy', 'bokeh', 'plotly', 'lasio', 'pywt', 'sklearn', 'xgboost']
missing_packages = [pkg for pkg in required_packages if not check_package(pkg)]
if missing_packages:
    st.error(f"Missing packages: {', '.join(missing_packages)}. Please install them using `pip install { ' '.join(missing_packages)}`")
    st.stop()

# Import rockphypy with error handling
try:
    from rockphypy import QI, GM, Fluid
    rockphypy_available = True
except ImportError:
    rockphypy_available = False
    st.warning("rockphypy package not available. RPT models will be disabled.")

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

# Presets for mineral properties
presets = {
    'Sandstone': {'rho_qz': 2.65, 'k_qz': 37.0, 'mu_qz': 44.0, 'rho_sh': 2.81, 'k_sh': 15.0, 'mu_sh': 5.0},
    'Carbonate': {'rho_qz': 2.71, 'k_qz': 76.8, 'mu_qz': 32.0, 'rho_sh': 2.81, 'k_sh': 15.0, 'mu_sh': 5.0}
}

# Rock Physics Model Functions
def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
    """Gassmann's Fluid Substitution"""
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1 * vs1**2
    k_s1 = rho1 * vp1**2 - (4./3.) * mu1
    kdry = (k_s1*((phi*k0)/k_f1 + 1 - phi) - k0) / \
           ((phi*k0)/k_f1 + (k_s1/k0) - 1 - phi)
    k_s2 = kdry + (1 - (kdry/k0))**2 / \
           ((phi/k_f2) + ((1-phi)/k0) - (kdry/k0**2))
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    mu2 = mu1
    vp2 = np.sqrt((k_s2 + (4./3)*mu2) / rho2)
    vs2 = np.sqrt(mu2 / rho2)
    return vp2*1000, vs2*1000, rho2, k_s2

def critical_porosity_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, phi_c):
    """Critical Porosity Model (Nur et al.)"""
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1*vs1**2.
    k_s1 = rho1*vp1**2 - (4./3.)*mu1
    kdry = k0 * (1 - phi/phi_c)
    mudry = mu0 * (1 - phi/phi_c)
    k_s2 = kdry + (1-(kdry/k0))**2/((phi/k_f2)+((1-phi)/k0)-(kdry/k0**2))
    rho2 = rho1-phi*rho_f1+phi*rho_f2
    mu2 = mudry
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2)
    vs2 = np.sqrt(mu2/rho2)
    return vp2*1000, vs2*1000, rho2, k_s2

def hertz_mindlin_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P):
    """Hertz-Mindlin contact theory model"""
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1*vs1**2.
    k_s1 = rho1*vp1**2 - (4./3.)*mu1
    PR0 = (3*k0 - 2*mu0)/(6*k0 + 2*mu0)
    kdry = (Cn**2 * (1 - phi)**2 * P * mu0**2 / (18 * np.pi**2 * (1 - PR0)**2))**(1/3)
    mudry = ((2 + 3*PR0 - PR0**2)/(5*(2 - PR0))) * (
        (3*Cn**2 * (1 - phi)**2 * P * mu0**2)/(2 * np.pi**2 * (1 - PR0)**2)
    )**(1/3)
    k_s2 = kdry + (1-(kdry/k0))**2/((phi/k_f2)+((1-phi)/k0)-(kdry/k0**2))
    rho2 = rho1-phi*rho_f1+phi*rho_f2
    mu2 = mudry
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2)
    vs2 = np.sqrt(mu2/rho2)
    return vp2*1000, vs2*1000, rho2, k_s2

def dvorkin_nur_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn=9, P=10, phi_c=0.4):
    """Dvorkin-Nur Soft Sand model"""
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    PR0 = (3*k0 - 2*mu0)/(6*k0 + 2*mu0)
    k_hm = (Cn**2 * (1-phi_c)**2 * P * mu0**2 / (18 * np.pi**2 * (1-PR0)**2))**(1/3)
    mu_hm = ((2 + 3*PR0 - PR0**2)/(5*(2-PR0))) * (
        (3*Cn**2 * (1-phi_c)**2 * P * mu0**2)/(2*np.pi**2*(1-PR0)**2)
    )**(1/3)
    k_dry = (phi/phi_c)/(k_hm + (4/3)*mu_hm) + (1 - phi/phi_c)/(k0 + (4/3)*mu_hm)
    k_dry = 1/k_dry - (4/3)*mu_hm
    k_dry = np.maximum(k_dry, 0)
    mu_dry = (phi/phi_c)/(mu_hm + (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm))) + \
             (1 - phi/phi_c)/(mu0 + (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm)))
    mu_dry = 1/mu_dry - (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm))
    mu_dry = np.maximum(mu_dry, 0)
    k_sat = k_dry + (1 - (k_dry/k0))**2 / ((phi/k_f2) + ((1-phi)/k0) - (k_dry/k0**2))
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    vp2 = np.sqrt((k_sat + (4/3)*mu_dry)/rho2) * 1000
    vs2 = np.sqrt(mu_dry/rho2) * 1000
    return vp2, vs2, rho2, k_sat

def raymer_hunt_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
    """Raymer-Hunt-Gardner empirical model"""
    vp_dry = (1 - phi)**2 * np.sqrt(k0/rho1) + phi * np.sqrt(k_f1/rho_f1)
    vp_dry = vp_dry * 1000
    vp_sat = (1 - phi)**2 * np.sqrt(k0/rho1) + phi * np.sqrt(k_f2/rho_f2)
    vp_sat = vp_sat * 1000
    vs_sat = vs1 * (1 - 1.5*phi)
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    return vp_sat, vs_sat, rho2, None

def xu_payne_model(vp1, vs1, rho1, phi, vsh, c=1.0, k0=37.0, mu0=44.0, k_sh=15.0, mu_sh=5.0):
    """Xu-Payne laminated sand-shale model"""
    k_qz = 37.0
    mu_qz = 44.0
    f_sh = vsh / (vsh + (1 - vsh) * c)
    k_dry = ((1 - f_sh) / (k_qz + (4/3)*mu_qz) + f_sh / (k_sh + (4/3)*mu_qz))**(-1) - (4/3)*mu_qz
    mu_dry = ((1 - f_sh) / (mu_qz + mu_qz/6*((9*k_qz + 8*mu_qz)/(k_qz + 2*mu_qz))) + 
              f_sh / (mu_sh + mu_qz/6*((9*k_qz + 8*mu_qz)/(k_qz + 2*mu_qz))))**(-1) - \
             mu_qz/6*((9*k_qz + 8*mu_qz)/(k_qz + 2*mu_qz))
    k_fl = 2.8
    k_sat = k_dry + (1 - k_dry/k0)**2 / (phi/k_fl + (1-phi)/k0 - k_dry/k0**2)
    rho2 = rho1 * (1 - phi) + phi * 1.0
    vp2 = np.sqrt((k_sat + (4/3)*mu_dry)/rho2) * 1000
    vs2 = np.sqrt(mu_dry/rho2) * 1000
    return vp2, vs2, rho2, k_sat

def greenberg_castagna(vp, vs, rho, phi, sw, lithology='sandstone'):
    """Greenberg-Castagna empirical Vp-Vs relationships"""
    coefficients = {
        'sandstone': {'a': 0.8042, 'b': -0.8559},
        'shale': {'a': 0.7697, 'b': -0.8674},
        'carbonate': {'a': 0.8535, 'b': -1.1375},
        'dolomite': {'a': 0.7825, 'b': -0.5529}
    }
    coeff = coefficients.get(lithology, coefficients['sandstone'])
    vp_km = vp / 1000
    vs_pred_km = coeff['a'] * vp_km + coeff['b']
    vs_pred = vs_pred_km * 1000
    vs_corr = vs_pred * (1 - 1.5*phi) * (1 - 0.5*sw)
    return vp, vs_corr, rho, None

# AVO and Seismic Modeling Functions
def ricker_wavelet(frequency, length=0.128, dt=0.001):
    """Generate a Ricker wavelet"""
    t = np.linspace(-length/2, length/2, int(length/dt))
    y = (1 - 2*(np.pi**2)*(frequency**2)*(t**2)) * np.exp(-(np.pi**2)*(frequency**2)*(t**2))
    return t, y

def smith_gidlow(vp1, vp2, vs1, vs2, rho1, rho2):
    """Calculate Smith-Gidlow AVO attributes"""
    rp = 0.5 * (vp2 - vp1) / (vp2 + vp1) + 0.5 * (rho2 - rho1) / (rho2 + rho1)
    rs = 0.5 * (vs2 - vs1) / (vs2 + vs1) + 0.5 * (rho2 - rho1) / (rho2 + rho1)
    intercept = rp
    gradient = rp - 2 * rs
    fluid_factor = rp + 1.16 * (vp1/vs1) * rs
    return intercept, gradient, fluid_factor

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

# Wedge Modeling Functions
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

def plot_vawig(axhdl, data, t, excursion, highlight=None):
    """Plot variable area wiggle traces"""
    [ntrc, nsamp] = data.shape
    t = np.hstack([0, t, t.max()])
    for i in range(0, ntrc):
        tbuf = excursion * data[i] / np.max(np.abs(data)) + i
        tbuf = np.hstack([i, tbuf, i])
        lw = 2 if i == highlight else 0.5
        axhdl.plot(tbuf, t, color='black', linewidth=lw)
        plt.fill_betweenx(t, tbuf, i, where=tbuf>i, facecolor=[0.6,0.6,1.0], linewidth=0)
        plt.fill_betweenx(t, tbuf, i, where=tbuf<i, facecolor=[1.0,0.7,0.7], linewidth=0)
    axhdl.set_xlim((-excursion, ntrc+excursion))
    axhdl.xaxis.tick_top()
    axhdl.xaxis.set_label_position('top')
    axhdl.invert_yaxis()

# Helper Functions
def monte_carlo_iteration(params, logs, model_func):
    """Single iteration of Monte Carlo simulation"""
    perturbed_params = {}
    for param, (mean, std) in params.items():
        perturbed_params[param] = np.random.normal(mean, std) if std > 0 else mean
    vp, vs, rho, _ = model_func(**perturbed_params)
    ip = vp * rho
    vpvs = vp / vs
    vp_upper = logs.VP.mean()
    vs_upper = logs.VS.mean()
    rho_upper = logs.RHO.mean()
    intercept, gradient, fluid_factor = smith_gidlow(vp_upper, vp, vs_upper, vs, rho_upper, rho)
    return {
        'VP': vp, 'VS': vs, 'RHO': rho,
        'IP': ip, 'VPVS': vpvs,
        'Intercept': intercept, 'Gradient': gradient, 'Fluid_Factor': fluid_factor
    }

def parallel_monte_carlo(logs, model_func, params, iterations=100):
    """Perform parallel Monte Carlo simulation"""
    results = {
        'VP': [], 'VS': [], 'RHO': [], 
        'IP': [], 'VPVS': [], 'Intercept': [], 
        'Gradient': [], 'Fluid_Factor': []
    }
    args = [(i, params) for i in range(iterations)]
    progress_bar = st.progress(0)
    status_text = st.empty()
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(0, iterations, 10):
            chunk = args[i:i+10]
            futures.append(executor.submit(process_monte_carlo_chunk, logs, model_func, chunk))
            progress = min((i + 10) / iterations, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing {int(progress*100)}% complete...")
        for future in futures:
            chunk_results = future.result()
            for key in results:
                results[key].extend(chunk_results[key])
    status_text.text("Monte Carlo simulation complete!")
    return results

def process_monte_carlo_chunk(logs, model_func, args):
    """Process a chunk of Monte Carlo iterations"""
    chunk_results = {
        'VP': [], 'VS': [], 'RHO': [], 
        'IP': [], 'VPVS': [], 'Intercept': [], 
        'Gradient': [], 'Fluid_Factor': []
    }
    for i, params in args:
        result = monte_carlo_iteration(params, logs, model_func)
        for key in chunk_results:
            chunk_results[key].append(result[key])
    return chunk_results

@st.cache_data
def create_interactive_crossplot(logs, depth_range=None):
    """Create interactive Bokeh crossplot"""
    if depth_range:
        logs = logs[(logs['DEPTH'] >= depth_range[0]) & (logs['DEPTH'] <= depth_range[1])]
    logs['LFC_MIX'] = logs['LFC_MIX'].astype(str)
    source = ColumnDataSource(logs)
    palette = Category10[10]
    p = figure(tools="pan,wheel_zoom,box_zoom,reset,hover,save",
               title="Interactive Crossplot")
    factors = sorted(logs['LFC_MIX'].unique())
    cmap = factor_cmap('LFC_MIX', palette=palette, factors=factors)
    p.scatter('IP_FRMMIX', 'VPVS_FRMMIX', size=8, source=source,
              color=cmap, alpha=0.6, legend_field='LFC_MIX')
    return p

def create_interactive_3d_crossplot(logs, x_col='IP', y_col='VPVS', z_col='RHO', color_col='LFC_B'):
    """Create interactive 3D crossplot with Plotly"""
    color_map = {
        0: 'gray', 1: 'blue', 2: 'green', 3: 'red', 4: 'magenta', 5: 'brown'
    }
    fig = go.Figure()
    for class_val, color in color_map.items():
        mask = logs[color_col] == class_val
        if mask.any():
            fig.add_trace(go.Scatter3d(
                x=logs.loc[mask, x_col],
                y=logs.loc[mask, y_col],
                z=logs.loc[mask, z_col],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.7, line=dict(width=0)),
                name=f'Class {class_val}',
                hovertemplate=
                    f"<b>{x_col}</b>: %{{x:.2f}}<br>" +
                    f"<b>{y_col}</b>: %{{y:.2f}}<br>" +
                    f"<b>{z_col}</b>: %{{z:.2f}}<br>" +
                    "<extra></extra>"
            ))
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
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5)
    )
    return fig

@st.cache_data
def predict_sonic_logs(logs, features, target_vp='VP', target_vs='VS', model_type='Random Forest'):
    """Predict VP and VS using selected ML model"""
    try:
        X = logs[features].dropna()
        y_vp = logs.loc[X.index, target_vp]
        y_vs = logs.loc[X.index, target_vs]
        X_train, X_test, y_vp_train, y_vp_test = train_test_split(
            X, y_vp, test_size=0.2, random_state=42)
        _, _, y_vs_train, y_vs_test = train_test_split(
            X, y_vs, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
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
        vp_model.fit(X_train_scaled, y_vp_train)
        vs_model.fit(X_train_scaled, y_vs_train)
        vp_pred = vp_model.predict(X_test_scaled)
        vs_pred = vs_model.predict(X_test_scaled)
        vp_r2 = r2_score(y_vp_test, vp_pred)
        vs_r2 = r2_score(y_vs_test, vs_pred)
        vp_rmse = np.sqrt(mean_squared_error(y_vp_test, vp_pred))
        vs_rmse = np.sqrt(mean_squared_error(y_vs_test, vs_pred))
        feature_importance = None
        if hasattr(vp_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': features,
                'VP_Importance': vp_model.feature_importances_,
                'VS_Importance': vs_model.feature_importances_
            })
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
        st.error(f"Sonic prediction failed: {str(e)}")
        logging.error(f"Sonic prediction error: {str(e)}")
        return None

@st.cache_data
def seismic_inversion_feasibility(logs, wavelet_freq):
    """Analyze seismic inversion feasibility"""
    try:
        results = {}
        nyquist = 0.5 / (logs.DEPTH.diff().mean()/logs.VP.mean())
        wavelet_bandwidth = wavelet_freq * 2
        results['bandwidth'] = {
            'nyquist': nyquist,
            'wavelet': wavelet_bandwidth,
            'feasible': wavelet_bandwidth < nyquist
        }
        corr_matrix = logs[['IP', 'VP', 'VS', 'RHO', 'VSH', 'PHI']].corr()
        sensitivity = {
            'VP_to_PHI': np.corrcoef(logs.VP, logs.PHI)[0,1],
            'VS_to_VSH': np.corrcoef(logs.VS, logs.VSH)[0,1],
            'IP_to_PHI': np.corrcoef(logs.IP, logs.PHI)[0,1]
        }
        tuning_thickness = logs.VP.mean() / (4 * wavelet_freq)
        return {
            'bandwidth': results['bandwidth'],
            'correlation_matrix': corr_matrix,
            'sensitivity': sensitivity,
            'tuning_thickness': tuning_thickness,
            'resolution': {
                'vertical': tuning_thickness,
                'horizontal': tuning_thickness * 2
            }
        }
    except Exception as e:
        st.error(f"Inversion feasibility analysis failed: {str(e)}")
        logging.error(f"Inversion feasibility error: {str(e)}")
        return None

def get_table_download_link(df, filename="results.csv"):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

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
    try:
        if not rockphypy_available:
            st.error("rockphypy package not available")
            return
        plt.figure(figsize=(10, 8))
        phi = np.linspace(0.01, phi_c, 20)
        if "Soft Sand" in title:
            Kdry, Gdry = GM.softsand(k_qz, mu_qz, phi, phi_c, Cn, sigma, f=f)
        else:
            Kdry, Gdry = GM.stiffsand(k_qz, mu_qz, phi, phi_c, Cn, sigma, f=f)
        ccc = ['#B3B3B3','blue','green','red','magenta','#996633']
        cmap_facies = colors.ListedColormap(ccc[0:6], 'indexed')
        if fluid == 'gas':
            QI.plot_rpt(Kdry, Gdry, k_qz, rho_qz, k_b, rho_b, k_g, rho_g, phi, np.linspace(0,1,5))
        elif fluid == 'oil':
            QI.plot_rpt(Kdry, Gdry, k_qz, rho_qz, k_b, rho_b, k_o, rho_o, phi, np.linspace(0,1,5))
        else:
            K_mix = (k_o * so + k_g * sg) / (so + sg + 1e-10)
            D_mix = (rho_o * so + rho_g * sg) / (so + sg + 1e-10)
            QI.plot_rpt(Kdry, Gdry, k_qz, rho_qz, k_b, rho_b, K_mix, D_mix, phi, np.linspace(0,1,5))
        if 'logs' in globals() and all(col in logs.columns for col in ['K_DRY', 'G_DRY', 'LFC_MIX']):
            depth_mask = (logs['DEPTH'] >= ztop) & (logs['DEPTH'] <= zbot)
            k_dry = logs.loc[depth_mask, 'K_DRY'] / 1e9
            g_dry = logs.loc[depth_mask, 'G_DRY'] / 1e9
            lfc = logs.loc[depth_mask, 'LFC_MIX']
            plt.scatter(k_dry, g_dry, c=lfc, cmap=cmap_facies, s=50, 
                       edgecolors='k', alpha=0.7, vmin=0, vmax=5)
            cbar = plt.colorbar()
            cbar.set_label('Fluid Type')
            cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
            cbar.set_ticklabels(['Undef','Brine','Oil','Gas','Mixed','Shale'])
        plt.title(f"{title} - {fluid.capitalize()} Case")
        plt.xlabel("P-Impedance (km/s*g/cc)")
        plt.ylabel("Vp/Vs Ratio")
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close()
    except Exception as e:
        st.error(f"Error generating RPT plot: {str(e)}")
        logging.error(f"RPT plot error: {str(e)}")

# Main Data Processing Function
@st.cache_data
def process_data(uploaded_file, model_choice, include_uncertainty=False, mc_iterations=100, **kwargs):
    """Process input data with selected rock physics model"""
    try:
        if isinstance(uploaded_file, str):
            logs = pd.read_csv(uploaded_file)
        else:
            uploaded_file.seek(0)
            if uploaded_file.name.endswith('.las'):
                las = lasio.read(uploaded_file)
                logs = las.df()
                logs['DEPTH'] = logs.index
            else:
                logs = pd.read_csv(uploaded_file)
        required_columns = {'DEPTH', 'VP', 'VS', 'RHO', 'VSH', 'SW', 'PHI'}
        for col in required_columns:
            if col not in logs.columns:
                logs[col] = np.nan
                st.warning(f"Missing column {col}. Filling with NaN and interpolating.")
        logs = logs.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        if len(logs) > 10000:
            logs = logs.sample(frac=0.5, random_state=42)
            st.info("Large dataset detected. Downsampled to 50% for performance.")
        sw = kwargs.get('sw', 0.8)
        so = kwargs.get('so', 0.15)
        sg = kwargs.get('sg', 0.05)
        if model_choice in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
            for case in ['B', 'O', 'G', 'MIX']:
                logs[f'VP_FRM{case}'] = logs.VP
                logs[f'VS_FRM{case}'] = logs.VS
                logs[f'RHO_FRM{case}'] = logs.RHO
                logs[f'IP_FRM{case}'] = logs.VP * logs.RHO
                logs[f'VPVS_FRM{case}'] = logs.VP/logs.VS
                logs[f'LFC_{case}'] = 0
            return logs, None
        rho_qz = kwargs.get('rho_qz', 2.65)
        k_qz = kwargs.get('k_qz', 37.0)
        mu_qz = kwargs.get('mu_qz', 44.0)
        rho_sh = kwargs.get('rho_sh', 2.81)
        k_sh = kwargs.get('k_sh', 15.0)
        mu_sh = kwargs.get('mu_sh', 5.0)
        rho_b = kwargs.get('rho_b', 1.09)
        k_b = kwargs.get('k_b', 2.8)
        rho_o = kwargs.get('rho_o', 0.78)
        k_o = kwargs.get('k_o', 0.94)
        rho_g = kwargs.get('rho_g', 0.25)
        k_g = kwargs.get('k_g', 0.06)
        sand_cutoff = kwargs.get('sand_cutoff', 0.12)
        rho_b_std = kwargs.get('rho_b_std', 0.05)
        k_b_std = kwargs.get('k_b_std', 0.1)
        rho_o_std = kwargs.get('rho_o_std', 0.05)
        k_o_std = kwargs.get('k_o_std', 0.05)
        rho_g_std = kwargs.get('rho_g_std', 0.02)
        k_g_std = kwargs.get('k_g_std', 0.01)
        def vrh(volumes, k, mu):
            f = np.array(volumes).T
            k = np.resize(np.array(k), np.shape(f))
            mu = np.resize(np.array(mu), np.shape(f))
            k_u = np.sum(f*k, axis=1)
            k_l = 1. / np.sum(f/k, axis=1)
            mu_u = np.sum(f*mu, axis=1)
            mu_l = 1. / np.sum(f/mu, axis=1)
            k0 = (k_u+k_l)/2.
            mu0 = (mu_u+mu_l)/2.
            return k_u, k_l, mu_u, mu_l, k0, mu0
        shale = logs.VSH.values
        sand = 1 - shale - logs.PHI.values
        shaleN = shale/(shale+sand)
        sandN = sand/(shale+sand)
        k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])
        water = sw
        oil = so
        gas = sg
        rho_fl = water*rho_b + oil*rho_o + gas*rho_g
        k_fl = 1.0 / (water/k_b + oil/k_o + gas/k_g)
        if model_choice == "Gassmann's Fluid Substitution":
            def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
                return frm(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi)
        elif model_choice == "Critical Porosity Model (Nur)":
            def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, phi_c):
                return critical_porosity_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, phi_c)
        elif model_choice == "Contact Theory (Hertz-Mindlin)":
            def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P):
                return hertz_mindlin_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P)
        elif model_choice == "Dvorkin-Nur Soft Sand Model":
            def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P, phi_c):
                return dvorkin_nur_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P, phi_c)
        elif model_choice == "Raymer-Hunt-Gardner Model":
            def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
                return raymer_hunt_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi)
        elif model_choice == "Xu-Payne Laminated Model":
            def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, vsh, c):
                return xu_payne_model(logs.VP, logs.VS, logs.RHO, phi, logs.VSH, c, k0, mu0, k_sh, mu_sh)
        elif model_choice == "Greenberg-Castagna Empirical":
            def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, sw, lithology):
                return greenberg_castagna(logs.VP, logs.VS, logs.RHO, phi, logs.SW, lithology)
        if model_choice == "Gassmann's Fluid Substitution":
            vpb, vsb, rhob, kb = model_func(rho_b, k_b, rho_b, k_b, k0, mu0, logs.PHI)
            vpo, vso, rhoo, ko = model_func(rho_b, k_b, rho_o, k_o, k0, mu0, logs.PHI)
            vpg, vsg, rhog, kg = model_func(rho_b, k_b, rho_g, k_g, k0, mu0, logs.PHI)
            vp_mix, vs_mix, rho_mix, k_mix = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI)
        elif model_choice == "Critical Porosity Model (Nur)":
            critical_porosity = kwargs.get('critical_porosity', 0.4)
            vpb, vsb, rhob, kb = model_func(rho_b, k_b, rho_b, k_b, k0, mu0, logs.PHI, critical_porosity)
            vpo, vso, rhoo, ko = model_func(rho_b, k_b, rho_o, k_o, k0, mu0, logs.PHI, critical_porosity)
            vpg, vsg, rhog, kg = model_func(rho_b, k_b, rho_g, k_g, k0, mu0, logs.PHI, critical_porosity)
            vp_mix, vs_mix, rho_mix, k_mix = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI, critical_porosity)
        elif model_choice == "Contact Theory (Hertz-Mindlin)":
            coordination_number = kwargs.get('coordination_number', 9)
            effective_pressure = kwargs.get('effective_pressure', 10)
            vpb, vsb, rhob, kb = model_func(rho_b, k_b, rho_b, k_b, k0, mu0, logs.PHI, coordination_number, effective_pressure)
            vpo, vso, rhoo, ko = model_func(rho_b, k_b, rho_o, k_o, k0, mu0, logs.PHI, coordination_number, effective_pressure)
            vpg, vsg, rhog, kg = model_func(rho_b, k_b, rho_g, k_g, k0, mu0, logs.PHI, coordination_number, effective_pressure)
            vp_mix, vs_mix, rho_mix, k_mix = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI, coordination_number, effective_pressure)
        elif model_choice == "Dvorkin-Nur Soft Sand Model":
            coordination_number = kwargs.get('coordination_number', 9)
            effective_pressure = kwargs.get('effective_pressure', 10)
            critical_porosity = kwargs.get('critical_porosity', 0.4)
            vpb, vsb, rhob, kb = model_func(rho_b, k_b, rho_b, k_b, k0, mu0, logs.PHI, coordination_number, effective_pressure, critical_porosity)
            vpo, vso, rhoo, ko = model_func(rho_b, k_b, rho_o, k_o, k0, mu0, logs.PHI, coordination_number, effective_pressure, critical_porosity)
            vpg, vsg, rhog, kg = model_func(rho_b, k_b, rho_g, k_g, k0, mu0, logs.PHI, coordination_number, effective_pressure, critical_porosity)
            vp_mix, vs_mix, rho_mix, k_mix = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI, coordination_number, effective_pressure, critical_porosity)
        elif model_choice == "Raymer-Hunt-Gardner Model":
            vpb, vsb, rhob, kb = model_func(rho_b, k_b, rho_b, k_b, k0, mu0, logs.PHI)
            vpo, vso, rhoo, ko = model_func(rho_b, k_b, rho_o, k_o, k0, mu0, logs.PHI)
            vpg, vsg, rhog, kg = model_func(rho_b, k_b, rho_g, k_g, k0, mu0, logs.PHI)
            vp_mix, vs_mix, rho_mix, k_mix = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI)
        elif model_choice == "Xu-Payne Laminated Model":
            lamination_factor = kwargs.get('lamination_factor', 1.0)
            vpb, vsb, rhob, kb = model_func(rho_b, k_b, rho_b, k_b, k0, mu0, logs.PHI, logs.VSH, lamination_factor)
            vpo, vso, rhoo, ko = model_func(rho_b, k_b, rho_o, k_o, k0, mu0, logs.PHI, logs.VSH, lamination_factor)
            vpg, vsg, rhog, kg = model_func(rho_b, k_b, rho_g, k_g, k0, mu0, logs.PHI, logs.VSH, lamination_factor)
            vp_mix, vs_mix, rho_mix, k_mix = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI, logs.VSH, lamination_factor)
        elif model_choice == "Greenberg-Castagna Empirical":
            lithology_type = kwargs.get('lithology_type', 'sandstone')
            vpb, vsb, rhob, kb = model_func(rho_b, k_b, rho_b, k_b, k0, mu0, logs.PHI, logs.SW, lithology_type)
            vpo, vso, rhoo, ko = model_func(rho_b, k_b, rho_o, k_o, k0, mu0, logs.PHI, logs.SW, lithology_type)
            vpg, vsg, rhog, kg = model_func(rho_b, k_b, rho_g, k_g, k0, mu0, logs.PHI, logs.SW, lithology_type)
            vp_mix, vs_mix, rho_mix, k_mix = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI, logs.SW, lithology_type)
        brine_sand = ((logs.VSH <= sand_cutoff) & (logs.SW >= 0.65))
        oil_sand = ((logs.VSH <= sand_cutoff) & (logs.SW < 0.65) & (logs.SW >= 0.35))
        gas_sand = ((logs.VSH <= sand_cutoff) & (logs.SW < 0.35))
        shale_flag = (logs.VSH > sand_cutoff)
        for case, vp, vs, rho in [('B', vpb, vsb, rhob), ('O', vpo, vso, rhoo), ('G', vpg, vsg, rhog), ('MIX', vp_mix, vs_mix, rho_mix)]:
            logs[f'VP_FRM{case}'] = logs.VP
            logs[f'VS_FRM{case}'] = logs.VS
            logs[f'RHO_FRM{case}'] = logs.RHO
            logs[f'VP_FRM{case}'][brine_sand|oil_sand|gas_sand] = vp[brine_sand|oil_sand|gas_sand]
            logs[f'VS_FRM{case}'][brine_sand|oil_sand|gas_sand] = vs[brine_sand|oil_sand|gas_sand]
            logs[f'RHO_FRM{case}'][brine_sand|oil_sand|gas_sand] = rho[brine_sand|oil_sand|gas_sand]
            logs[f'IP_FRM{case}'] = logs[f'VP_FRM{case}']*logs[f'RHO_FRM{case}']
            logs[f'IS_FRM{case}'] = logs[f'VS_FRM{case}']*logs[f'RHO_FRM{case}']
            logs[f'VPVS_FRM{case}'] = logs[f'VP_FRM{case}']/logs[f'VS_FRM{case}']
        for case, val in [('B', 1), ('O', 2), ('G', 3), ('MIX', 4)]:
            temp_lfc = np.zeros(np.shape(logs.VSH))
            temp_lfc[brine_sand.values | oil_sand.values | gas_sand.values] = val
            temp_lfc[shale_flag.values] = 5
            logs[f'LFC_{case}'] = temp_lfc
        mc_results = None
        if include_uncertainty:
            params = {
                'rho_f1': (rho_b, rho_b_std),
                'k_f1': (k_b, k_b_std),
                'rho_f2': (rho_fl, np.sqrt((sw*rho_b_std)**2 + (so*rho_o_std)**2 + (sg*rho_g_std)**2)),
                'k_f2': (k_fl, np.sqrt((sw*k_b_std)**2 + (so*k_o_std)**2 + (sg*k_g_std)**2)),
                'k0': (k0.mean(), 0.1 * k0.mean()),
                'mu0': (mu0.mean(), 0.1 * mu0.mean()),
                'phi': (logs.PHI.mean(), 0.05)
            }
            if model_choice == "Critical Porosity Model (Nur)":
                params['phi_c'] = (critical_porosity, 0.01)
            elif model_choice in ["Contact Theory (Hertz-Mindlin)", "Dvorkin-Nur Soft Sand Model"]:
                params['Cn'] = (coordination_number, 1)
                params['P'] = (effective_pressure, 5)
                if model_choice == "Dvorkin-Nur Soft Sand Model":
                    params['phi_c'] = (critical_porosity, 0.01)
            elif model_choice == "Xu-Payne Laminated Model":
                params['vsh'] = (logs.VSH.mean(), 0.05)
                params['c'] = (lamination_factor, 0.1)
            elif model_choice == "Greenberg-Castagna Empirical":
                params['sw'] = (logs.SW.mean(), 0.05)
                params['lithology'] = lithology_type
            mc_results = parallel_monte_carlo(logs, model_func, params, min(mc_iterations, 500))
        return logs, mc_results
    except Exception as e:
        st.error(f"Data processing failed: {str(e)}")
        logging.error(f"Data processing error: {str(e)}")
        return None, None

# Sidebar for Input Parameters
with st.sidebar:
    st.header("Model Configuration")
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
        model_options.extend(["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"])
    model_choice = st.selectbox("Rock Physics Model", model_options, index=0,
                                help="Select the rock physics model for analysis")
    with st.expander("Mineral Properties"):
        preset = st.selectbox("Select Preset", ["None"] + list(presets.keys()), key="preset")
        col1, col2 = st.columns(2)
        with col1:
            rho_qz = st.number_input("Quartz Density (g/cc)", value=presets[preset]['rho_qz'] if preset != "None" else 2.65, 
                                    step=0.01, key="rho_qz")
            k_qz = st.number_input("Quartz Bulk Modulus (GPa)", value=presets[preset]['k_qz'] if preset != "None" else 37.0, 
                                   step=0.1, key="k_qz")
            mu_qz = st.number_input("Quartz Shear Modulus (GPa)", value=presets[preset]['mu_qz'] if preset != "None" else 44.0, 
                                    step=0.1, key="mu_qz")
        with col2:
            rho_sh = st.number_input("Shale Density (g/cc)", value=presets[preset]['rho_sh'] if preset != "None" else 2.81, 
                                    step=0.01, key="rho_sh")
            k_sh = st.number_input("Shale Bulk Modulus (GPa)", value=presets[preset]['k_sh'] if preset != "None" else 15.0, 
                                   step=0.1, key="k_sh")
            mu_sh = st.number_input("Shale Shear Modulus (GPa)", value=presets[preset]['mu_sh'] if preset != "None" else 5.0, 
                                    step=0.1, key="mu_sh")
    if model_choice == "Critical Porosity Model (Nur)":
        critical_porosity = st.slider("Critical Porosity (φc)", 0.3, 0.5, 0.4, 0.01, key="phi_c",
                                     help="Porosity at which the rock loses mechanical competence")
    elif model_choice in ["Contact Theory (Hertz-Mindlin)", "Dvorkin-Nur Soft Sand Model"]:
        coordination_number = st.slider("Coordination Number", 6, 12, 9, key="Cn",
                                       help="Average number of grain contacts")
        effective_pressure = st.slider("Effective Pressure (MPa)", 1, 50, 10, key="P",
                                      help="Effective stress acting on the rock")
        if model_choice == "Dvorkin-Nur Soft Sand Model":
            critical_porosity = st.slider("Critical Porosity (φc)", 0.3, 0.5, 0.4, 0.01, key="phi_c_sand")
    elif model_choice == "Xu-Payne Laminated Model":
        lamination_factor = st.slider("Lamination Factor (c)", 0.1, 2.0, 1.0, 0.1, key="c",
                                     help="Controls sand-shale lamination")
    elif model_choice == "Greenberg-Castagna Empirical":
        lithology_type = st.selectbox("Lithology Type", ['sandstone', 'shale', 'carbonate', 'dolomite'], key="lithology")
    if model_choice in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
        st.subheader("RPT Model Parameters")
        rpt_phi_c = st.slider("RPT Critical Porosity", 0.3, 0.5, 0.4, 0.01, key="rpt_phi_c")
        rpt_Cn = st.slider("RPT Coordination Number", 6.0, 12.0, 8.6, 0.1, key="rpt_Cn")
        rpt_sigma = st.slider("RPT Effective Stress (MPa)", 1, 50, 20, key="rpt_sigma")
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
    with st.expander("Saturation Settings"):
        sw_default = 0.8
        so_default = 0.15
        sg_default = 0.05
        sw = st.slider("Water Saturation (Sw)", 0.0, 1.0, sw_default, 0.01, key="sw")
        remaining = max(0.0, 1.0 - sw)
        so = st.slider("Oil Saturation (So)", 0.0, remaining, min(so_default, remaining), 0.01, key="so")
        sg = remaining - so
        st.write(f"Current saturations: Sw={sw:.2f}, So={so:.2f}, Sg={sg:.2f}")
    with st.expander("AVO Modeling Parameters"):
        min_angle = st.slider("Minimum Angle (deg)", 0, 10, 0, key="avo_min_angle")
        max_angle = st.slider("Maximum Angle (deg)", 30, 50, 45, key="avo_max_angle")
        angle_step = st.slider("Angle Step (deg)", 1, 5, 1, key="angle_step")
        wavelet_freq = st.slider("Wavelet Frequency (Hz)", 20, 80, 50, key="wavelet_freq")
        sand_cutoff = st.slider("Sand Cutoff (VSH)", 0.0, 0.3, 0.12, step=0.01, key="sand_cutoff")
    with st.expander("Time-Frequency Analysis"):
        cwt_scales = st.slider("CWT Scales Range", 1, 100, (1, 50), key="cwt_scales")
        cwt_wavelet = st.selectbox("CWT Wavelet", ['morl', 'cmor', 'gaus', 'mexh'], index=0, key="cwt_wavelet")
    with st.expander("Uncertainty Analysis"):
        mc_iterations = st.slider("Monte Carlo Iterations", 10, 500, 100, key="mc_iterations")
        parallel_processing = st.checkbox("Use parallel processing", value=True, key="parallel_processing")
        include_uncertainty = st.checkbox("Include Uncertainty Analysis", value=False, key="include_uncertainty")
    with st.expander("Visualization Options"):
        selected_cmap = st.selectbox("Color Map", seismic_colormaps, index=0, key="colormap")
        show_3d_crossplot = st.checkbox("Show 3D Crossplot", value=False, key="show_3d_crossplot")
        show_interactive_3d = st.checkbox("Show Interactive 3D Plot", value=True, key="show_interactive_3d")
        show_histograms = st.checkbox("Show Histograms", value=True, key="show_histograms")
        show_smith_gidlow = st.checkbox("Show Smith-Gidlow AVO Attributes", value=True, key="show_smith_gidlow")
        show_wedge_model = st.checkbox("Show Wedge Modeling", value=False, key="show_wedge_model")
    with st.expander("Advanced Modules"):
        predict_sonic = st.checkbox("Enable Sonic Log Prediction", value=False, key="predict_sonic")
        if predict_sonic:
            ml_model = st.selectbox("ML Model", 
                                  ["Random Forest", "XGBoost", "Gaussian Process", "Neural Network"],
                                  index=0, key="ml_model")
        inversion_feasibility = st.checkbox("Enable Seismic Inversion Feasibility", value=False, key="inversion_feasibility")
    with st.expander("Data Preprocessing"):
        remove_outliers = st.checkbox("Remove Outliers", value=False, key="remove_outliers")
        uploaded_file = st.file_uploader("Upload CSV or LAS file", type=["csv", "las"], key="file_uploader")
        if uploaded_file:
            st.success("File uploaded successfully!")

# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            logging.error(f"Error in {func.__name__}: {str(e)}")
            st.stop()
    return wrapper

# Main Application
if uploaded_file is not None:
    try:
        original_file = uploaded_file
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
            rho_b_std=rho_b_std, k_b_std=k_b_std,
            rho_o_std=rho_o_std, k_o_std=k_o_std,
            rho_g_std=rho_g_std, k_g_std=k_g_std,
            sand_cutoff=sand_cutoff,
            sw=sw, so=so, sg=sg,
            critical_porosity=critical_porosity if 'critical_porosity' in locals() else None,
            coordination_number=coordination_number if 'coordination_number' in locals() else None,
            effective_pressure=effective_pressure if 'effective_pressure' in locals() else None,
            lamination_factor=lamination_factor if 'lamination_factor' in locals() else None,
            lithology_type=lithology_type if 'lithology_type' in locals() else None
        )
        if logs is None:
            st.error("Data processing failed - check your input data")
            st.stop()
        if remove_outliers:
            for col in ['VP', 'VS', 'RHO', 'PHI']:
                if col in logs.columns:
                    logs = logs[logs[col].between(logs[col].quantile(0.01), logs[col].quantile(0.99))]
            st.success("Outliers removed successfully")
        ztop, zbot = st.slider(
            "Select Depth Range", 
            float(logs.DEPTH.min()), 
            float(logs.DEPTH.max()), 
            (float(logs.DEPTH.min()), float(logs.DEPTH.max())),
            key="depth_range"
        )
        if st.button("Export Results"):
            logs.to_csv("processed_logs.csv")
            st.markdown(get_table_download_link(logs), unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Well Log Visualization", 
            "Crossplots", 
            "AVO Analysis", 
            "Rock Physics Templates",
            "Wedge Modeling"
        ])
        ccc = ['#B3B3B3','blue','green','red','magenta','#996633']
        cmap_facies = colors.ListedColormap(ccc[0:len(ccc)], 'indexed')
        ll = logs.loc[(logs.DEPTH>=ztop) & (logs.DEPTH<=zbot)]
        cluster = np.repeat(np.expand_dims(ll['LFC_B'].values,1), 100, 1)
        with tab1:
            if model_choice not in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"]:
                st.header("Interactive Well Log Visualization")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ll.VSH, y=ll.DEPTH, mode='lines', name='Vsh', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=ll.SW, y=ll.DEPTH, mode='lines', name='Sw', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=ll.PHI, y=ll.DEPTH, mode='lines', name='Phi', line=dict(color='black')))
                fig.update_layout(
                    yaxis=dict(autorange='reversed', title='Depth (m)'),
                    xaxis=dict(title='Vcl/Phi/Sw', range=[-0.1, 1.1]),
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                fig_ip = go.Figure()
                fig_ip.add_trace(go.Scatter(x=ll.IP, y=ll.DEPTH, mode='lines', name='Original', line=dict(color='gray')))
                fig_ip.add_trace(go.Scatter(x=ll.IP_FRMB, y=ll.DEPTH, mode='lines', name='Brine', line=dict(color='blue')))
                fig_ip.add_trace(go.Scatter(x=ll.IP_FRMG, y=ll.DEPTH, mode='lines', name='Gas', line=dict(color='red')))
                fig_ip.add_trace(go.Scatter(x=ll.IP_FRMMIX, y=ll.DEPTH, mode='lines', name='Mixed', line=dict(color='magenta')))
                fig_ip.update_layout(
                    yaxis=dict(autorange='reversed', title='Depth (m)'),
                    xaxis=dict(title='IP (m/s*g/cc)', range=[6000, 15000]),
                    height=600
                )
                st.plotly_chart(fig_ip, use_container_width=True)
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
                            if prediction_results['feature_importance'] is not None:
                                st.subheader("Feature Importance")
                                fig_imp =
