import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import base64
from streamlit_bokeh_events import streamlit_bokeh_events
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CustomJS
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
import lasio

# Import rockphypy with error handling
try:
    from rockphypy import QI, GM, Fluid
    rockphypy_available = True
except ImportError:
    rockphypy_available = False

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced Rock Physics & AVO Modeling with Inversion Feasibility")

# Title and description
st.title("Enhanced Rock Physics & AVO Modeling Tool")
st.markdown("""
This app performs advanced rock physics modeling and AVO analysis with multiple models, visualization options, 
uncertainty analysis, sonic log prediction, and seismic inversion feasibility assessment.
""")

# Available colormaps for seismic displays
seismic_colormaps = ['seismic', 'RdBu', 'bwr', 'coolwarm', 'viridis', 'plasma']

# Sidebar for input parameters
with st.sidebar:
    st.header("Model Configuration")
    
    # Rock physics model selection
    model_options = [
        "Gassmann's Fluid Substitution", 
        "Critical Porosity Model (Nur)", 
        "Contact Theory (Hertz-Mindlin)",
        "Dvorkin-Nur Soft Sand Model",
        "Raymer-Hunt-Gardner Model"
    ]
    
    if rockphypy_available:
        model_options.extend([
            "Soft Sand RPT (rockphypy)",
            "Stiff Sand RPT (rockphypy)"
        ])
    
    model_choice = st.selectbox("Rock Physics Model", model_options, index=0)
    
    # Mineral properties
    st.subheader("Mineral Properties")
    col1, col2 = st.columns(2)
    with col1:
        rho_qz = st.number_input("Quartz Density (g/cc)", value=2.65, step=0.01)
        k_qz = st.number_input("Quartz Bulk Modulus (GPa)", value=37.0, step=0.1)
        mu_qz = st.number_input("Quartz Shear Modulus (GPa)", value=44.0, step=0.1)
    with col2:
        rho_sh = st.number_input("Shale Density (g/cc)", value=2.81, step=0.01)
        k_sh = st.number_input("Shale Bulk Modulus (GPa)", value=15.0, step=0.1)
        mu_sh = st.number_input("Shale Shear Modulus (GPa)", value=5.0, step=0.1)
    
    # Additional parameters for selected models
    if model_choice == "Critical Porosity Model (Nur)":
        critical_porosity = st.slider("Critical Porosity (φc)", 0.3, 0.5, 0.4, 0.01)
    elif model_choice in ["Contact Theory (Hertz-Mindlin)", "Dvorkin-Nur Soft Sand Model"]:
        coordination_number = st.slider("Coordination Number", 6, 12, 9)
        effective_pressure = st.slider("Effective Pressure (MPa)", 1, 50, 10)
        if model_choice == "Dvorkin-Nur Soft Sand Model":
            critical_porosity = st.slider("Critical Porosity (φc)", 0.3, 0.5, 0.4, 0.01)
    
    # Rockphypy specific parameters
    if model_choice in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"] and rockphypy_available:
        st.subheader("RPT Model Parameters")
        rpt_phi_c = st.slider("RPT Critical Porosity", 0.3, 0.5, 0.4, 0.01)
        rpt_Cn = st.slider("RPT Coordination Number", 6.0, 12.0, 8.6, 0.1)
        rpt_sigma = st.slider("RPT Effective Stress (MPa)", 1, 50, 20)
    
    # Fluid properties with uncertainty ranges
    st.subheader("Fluid Properties")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Brine**")
        rho_b = st.number_input("Brine Density (g/cc)", value=1.09, step=0.01)
        k_b = st.number_input("Brine Bulk Modulus (GPa)", value=2.8, step=0.1)
        rho_b_std = st.number_input("Brine Density Std Dev", value=0.05, step=0.01, min_value=0.0)
        k_b_std = st.number_input("Brine Bulk Modulus Std Dev", value=0.1, step=0.01, min_value=0.0)
    with col2:
        st.markdown("**Oil**")
        rho_o = st.number_input("Oil Density (g/cc)", value=0.78, step=0.01)
        k_o = st.number_input("Oil Bulk Modulus (GPa)", value=0.94, step=0.1)
        rho_o_std = st.number_input("Oil Density Std Dev", value=0.05, step=0.01, min_value=0.0)
        k_o_std = st.number_input("Oil Bulk Modulus Std Dev", value=0.05, step=0.01, min_value=0.0)
    with col3:
        st.markdown("**Gas**")
        rho_g = st.number_input("Gas Density (g/cc)", value=0.25, step=0.01)
        k_g = st.number_input("Gas Bulk Modulus (GPa)", value=0.06, step=0.01)
        rho_g_std = st.number_input("Gas Density Std Dev", value=0.02, step=0.01, min_value=0.0)
        k_g_std = st.number_input("Gas Bulk Modulus Std Dev", value=0.01, step=0.01, min_value=0.0)
    
    # Saturation controls
    st.subheader("Saturation Settings")
    sw_default = 0.8
    so_default = 0.15
    sg_default = 0.05
    
    sw = st.slider("Water Saturation (Sw)", 0.0, 1.0, sw_default, 0.01)
    remaining = max(0.0, 1.0 - sw)  # Ensure remaining is never negative
    so = st.slider(
        "Oil Saturation (So)", 
        0.0, 
        remaining, 
        min(so_default, remaining) if remaining > 0 else 0.0, 
        0.01
    )
    sg = remaining - so
    
    # Display actual saturations (in case adjustments were made)
    st.write(f"Current saturations: Sw={sw:.2f}, So={so:.2f}, Sg={sg:.2f}")
    
    # AVO modeling parameters
    st.subheader("AVO Modeling Parameters")
    min_angle = st.slider("Minimum Angle (deg)", 0, 10, 0)
    max_angle = st.slider("Maximum Angle (deg)", 30, 50, 45)
    angle_step = st.slider("Angle Step (deg)", 1, 5, 1)
    wavelet_freq = st.slider("Wavelet Frequency (Hz)", 20, 80, 50)
    sand_cutoff = st.slider("Sand Cutoff (VSH)", 0.0, 0.3, 0.12, step=0.01)
    
    # Time-Frequency Analysis Parameters
    st.subheader("Time-Frequency Analysis")
    cwt_scales = st.slider("CWT Scales Range", 1, 100, (1, 50))
    cwt_wavelet = st.selectbox("CWT Wavelet", ['morl', 'cmor', 'gaus', 'mexh'], index=0)
    
    # Monte Carlo parameters
    st.subheader("Uncertainty Analysis")
    mc_iterations = st.slider("Monte Carlo Iterations", 10, 1000, 100)
    include_uncertainty = st.checkbox("Include Uncertainty Analysis", value=False)
    
    # Visualization options
    st.subheader("Visualization Options")
    selected_cmap = st.selectbox("Color Map", seismic_colormaps, index=0)
    show_3d_crossplot = st.checkbox("Show 3D Crossplot", value=False)
    show_histograms = st.checkbox("Show Histograms", value=True)
    show_smith_gidlow = st.checkbox("Show Smith-Gidlow AVO Attributes", value=True)
    
    # Advanced Modules
    st.header("Advanced Modules")
    predict_sonic = st.checkbox("Enable Sonic Log Prediction", value=False)
    inversion_feasibility = st.checkbox("Enable Seismic Inversion Feasibility", value=False)
    
    # Wedge model parameters
    st.subheader("Wedge Model Settings")
    wedge_min = st.number_input("Minimum Wedge Thickness (m)", value=1.0, min_value=0.1, step=0.5)
    wedge_max = st.number_input("Maximum Wedge Thickness (m)", value=50.0, min_value=1.0, step=5.0)
    
    # File upload
    st.subheader("Well Log Data")
    uploaded_file = st.file_uploader("Upload CSV or LAS file", type=["csv", "las"])

# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            st.stop()
    return wrapper

# Rock Physics Models
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
           ((phi/k_f2) + ((1 - phi)/k0) - (kdry/k0**2))
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    mu2 = mu1  # Shear modulus unaffected by fluid change
    vp2 = np.sqrt((k_s2 + (4./3)*mu2) / rho2)
    vs2 = np.sqrt(mu2 / rho2)

    return vp2*1000, vs2*1000, rho2, k_s2

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
    mu2 = mudry  # Shear modulus not affected by fluid in Gassmann
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2)
    vs2 = np.sqrt(mu2/rho2)
    
    return vp2*1000, vs2*1000, rho2, k_s2

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

# Wavelet function
def ricker_wavelet(frequency, length=0.128, dt=0.001):
    """Generate a Ricker wavelet"""
    t = np.linspace(-length/2, length/2, int(length/dt))
    y = (1 - 2*(np.pi**2)*(frequency**2)*(t**2)) * np.exp(-(np.pi**2)*(frequency**2)*(t**2))
    return t, y

# Smith-Gidlow AVO approximation
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

# Calculate reflection coefficients
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

# Fit AVO curve to get intercept and gradient
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

# Monte Carlo simulation for uncertainty analysis
def monte_carlo_simulation(logs, model_func, params, iterations=100):
    """Perform Monte Carlo simulation for uncertainty analysis"""
    results = {
        'VP': [], 'VS': [], 'RHO': [], 
        'IP': [], 'VPVS': [], 'Intercept': [], 
        'Gradient': [], 'Fluid_Factor': []
    }
    
    for _ in range(iterations):
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
        
        # Store results
        results['VP'].append(vp)
        results['VS'].append(vs)
        results['RHO'].append(rho)
        results['IP'].append(ip)
        results['VPVS'].append(vpvs)
        results['Intercept'].append(intercept)
        results['Gradient'].append(gradient)
        results['Fluid_Factor'].append(fluid_factor)
    
    return results

# Create interactive crossplot with improved error handling
def create_interactive_crossplot(logs):
    """Create interactive Bokeh crossplot with proper error handling"""
    try:
        # Define litho-fluid class labels
        lfc_labels = ['Undefined', 'Brine', 'Oil', 'Gas', 'Mixed', 'Shale']
        
        # Handle NaN values and ensure integers
        if 'LFC_B' not in logs.columns:
            logs['LFC_B'] = 0
        logs['LFC_B'] = logs['LFC_B'].fillna(0).clip(0, 5).astype(int)
        
        # Create labels - handle any unexpected values gracefully
        logs['LFC_Label'] = logs['LFC_B'].apply(
            lambda x: lfc_labels[x] if x in range(len(lfc_labels)) else 'Undefined'
        )
        
        # Filter out NaN values and ensure numeric data
        plot_data = logs[['IP', 'VPVS', 'LFC_Label', 'DEPTH']].dropna()
        plot_data = plot_data[plot_data.apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull().all(axis=1)]
        
        if len(plot_data) == 0:
            st.warning("No valid data available for crossplot - check your input data")
            return None
            
        # Create ColumnDataSource
        source = ColumnDataSource(plot_data)
        
        # Get unique labels present in data
        unique_labels = sorted(plot_data['LFC_Label'].unique())
        
        # Create figure
        p = figure(width=800, height=500, 
                  tools="box_select,lasso_select,pan,wheel_zoom,box_zoom,reset",
                  title="IP vs Vp/Vs Crossplot")
        
        # Create color map based on actual labels present
        if len(unique_labels) > 0:
            color_map = factor_cmap('LFC_Label', 
                                  palette=Category10[len(unique_labels)], 
                                  factors=unique_labels)
            
            # Create scatter plot
            scatter = p.scatter('IP', 'VPVS', source=source, size=5,
                              color=color_map, legend_field='LFC_Label',
                              alpha=0.6)
            
            # Configure axes and legend
            p.xaxis.axis_label = 'IP (m/s*g/cc)'
            p.yaxis.axis_label = 'Vp/Vs'
            p.legend.title = 'Litho-Fluid Class'
            p.legend.location = "top_right"
            p.legend.click_policy = "hide"
            
            # Add hover tool
            hover = HoverTool(tooltips=[
                ("Depth", "@DEPTH{0.2f}"),
                ("IP", "@IP{0.2f}"),
                ("Vp/Vs", "@VPVS{0.2f}"),
                ("Class", "@LFC_Label")
            ])
            p.add_tools(hover)
            
            return p
        else:
            st.warning("No valid class labels found for crossplot")
            return None
            
    except Exception as e:
        st.error(f"Error creating interactive crossplot: {str(e)}")
        return None

# Sonic Log Prediction Function
def predict_vp_vs(logs, features):
    """Predict VP and VS using machine learning (Random Forest)"""
    try:
        # Select features and target
        X = logs[features].dropna()
        y_vp = logs.loc[X.index, 'VP']
        y_vs = logs.loc[X.index, 'VS']
        
        # Train-test split
        X_train, X_test, y_vp_train, y_vp_test = train_test_split(
            X, y_vp, test_size=0.2, random_state=42)
        _, _, y_vs_train, y_vs_test = train_test_split(
            X, y_vs, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        vp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        vs_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        vp_model.fit(X_train_scaled, y_vp_train)
        vs_model.fit(X_train_scaled, y_vs_train)
        
        # Predictions (on all data)
        X_scaled = scaler.transform(X)
        logs.loc[X.index, 'VP_pred'] = vp_model.predict(X_scaled)
        logs.loc[X.index, 'VS_pred'] = vs_model.predict(X_scaled)
        
        # Metrics
        vp_r2 = r2_score(y_vp_test, vp_model.predict(X_test_scaled))
        vs_r2 = r2_score(y_vs_test, vs_model.predict(X_test_scaled))
        vp_rmse = np.sqrt(mean_squared_error(y_vp_test, vp_model.predict(X_test_scaled)))
        vs_rmse = np.sqrt(mean_squared_error(y_vs_test, vs_model.predict(X_test_scaled)))
        
        # Plot predicted vs. original logs
        fig_pred, ax = plt.subplots(1, 2, figsize=(15, 10))
        
        # VP plot
        ax[0].plot(logs.VP, logs.DEPTH, 'k-', label='Original VP', linewidth=2)
        ax[0].plot(logs.VP_pred, logs.DEPTH, 'r--', label='Predicted VP', linewidth=1.5)
        ax[0].set_title("VP Prediction")
        ax[0].set_xlabel("VP (m/s)")
        ax[0].set_ylabel("Depth (m)")
        ax[0].legend()
        ax[0].grid(True)
        ax[0].invert_yaxis()
        
        # VS plot
        ax[1].plot(logs.VS, logs.DEPTH, 'k-', label='Original VS', linewidth=2)
        ax[1].plot(logs.VS_pred, logs.DEPTH, 'b--', label='Predicted VS', linewidth=1.5)
        ax[1].set_title("VS Prediction")
        ax[1].set_xlabel("VS (m/s)")
        ax[1].legend()
        ax[1].grid(True)
        ax[1].invert_yaxis()
        
        plt.tight_layout()
        st.pyplot(fig_pred)
        
        return {
            'vp_model': vp_model,
            'vs_model': vs_model,
            'scaler': scaler,
            'vp_r2': vp_r2,
            'vs_r2': vs_r2,
            'vp_rmse': vp_rmse,
            'vs_rmse': vs_rmse,
            'features': features
        }
    except Exception as e:
        st.error(f"Sonic prediction failed: {str(e)}")
        return None

# Wedge Model Function (Corrected)
def plot_wedge_model(vp_log, vs_log, rho_log, depth_log, wavelet_freq, thickness_range=(1, 50)):
    """
    Generate a seismic wedge model (3-layer: shale/sand/shale) with synthetic seismic.
    """
    # Create impedance log
    ip_log = vp_log * rho_log
    
    # Define layer indices (shale = background, sand = wedge)
    sand_thicknesses = np.linspace(thickness_range[0], thickness_range[1], 20)
    middle_depth = depth_log.mean()
    
    # Initialize wedge model (each row is a trace with varying sand thickness)
    wedge_model = np.zeros((len(sand_thicknesses), len(depth_log)))
    synthetic_seismic = np.zeros_like(wedge_model)
    
    # Generate Ricker wavelet
    wavelet_time, wavelet_amp = ricker_wavelet(wavelet_freq)
    wavelet_amp = wavelet_amp / np.max(np.abs(wavelet_amp))  # Normalize
    
    for i, thickness in enumerate(sand_thicknesses):
        # Create wedge layer
        sand_top = middle_depth - thickness / 2
        sand_bottom = middle_depth + thickness / 2
        sand_layer = (depth_log >= sand_top) & (depth_log <= sand_bottom)
        
        # Assign impedance values (shale = background, sand = wedge)
        wedge_model[i, :] = ip_log
        wedge_model[i, sand_layer] = ip_log[sand_layer] * 0.9  # Sand has lower IP
        
        # Create reflectivity series
        rc = np.zeros(len(depth_log))
        rc[1:] = (wedge_model[i, 1:] - wedge_model[i, :-1]) / (wedge_model[i, 1:] + wedge_model[i, :-1])
        
        # Convolve with wavelet to generate synthetic seismic
        synthetic_seismic[i, :] = np.convolve(rc, wavelet_amp, mode='same')
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Wedge model (impedance)
    im1 = ax1.imshow(
        wedge_model, 
        aspect='auto', 
        extent=[depth_log.min(), depth_log.max(), thickness_range[1], thickness_range[0]],
        cmap='viridis'
    )
    ax1.set_title("Impedance Wedge Model")
    ax1.set_xlabel("Depth (m)")
    ax1.set_ylabel("Wedge Thickness (m)")
    plt.colorbar(im1, ax=ax1, label="Acoustic Impedance (m/s·g/cc)")
    
    # Synthetic seismic
    im2 = ax2.imshow(
        synthetic_seismic, 
        aspect='auto', 
        extent=[depth_log.min(), depth_log.max(), thickness_range[1], thickness_range[0]],
        cmap='seismic',
        vmin=-np.max(np.abs(synthetic_seismic)), 
        vmax=np.max(np.abs(synthetic_seismic))
    )
    ax2.set_title(f"Synthetic Seismic (Wavelet: {wavelet_freq} Hz)")
    ax2.set_xlabel("Depth (m)")
    plt.colorbar(im2, ax=ax2, label="Amplitude")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Return synthetic seismic for frequency analysis
    return synthetic_seismic

# Frequency Analysis Functions
def plot_frequency_analysis(synthetic_seismic, time_samples, angles, wavelet_freq, time_range=(0.15, 0.25)):
    """
    Plot frequency spectrum and time-frequency analysis of synthetic seismic.
    """
    # Select a middle trace for analysis
    middle_trace = synthetic_seismic[len(synthetic_seismic) // 2, :]
    
    # Compute FFT
    n = len(middle_trace)
    dt = time_samples[1] - time_samples[0]
    freq = np.fft.fftfreq(n, dt)[:n//2]
    fft_vals = np.abs(np.fft.fft(middle_trace)[:n//2])
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Frequency spectrum
    ax1.plot(freq, fft_vals, 'b-', linewidth=2)
    ax1.axvline(wavelet_freq, color='r', linestyle='--', label=f'Wavelet Freq: {wavelet_freq} Hz')
    ax1.set_title("Frequency Spectrum")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)
    
    # Time-frequency analysis (STFT)
    f, t, Zxx = signal.stft(
        middle_trace,
        fs=1/dt,
        nperseg=64,
        noverlap=32
    )
    ax2.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
    ax2.set_title("Short-Time Fourier Transform (STFT)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_ylim(0, 2 * wavelet_freq)  # Focus on relevant frequencies
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_cwt_analysis(synthetic_seismic, time_samples, angles, scales, wavelet='morl'):
    """
    Plot Continuous Wavelet Transform (CWT) of synthetic seismic.
    """
    # Select a middle trace
    trace = synthetic_seismic[len(synthetic_seismic) // 2, :]
    
    # Compute CWT
    coefficients, frequencies = pywt.cwt(trace, scales, wavelet, sampling_period=time_samples[1]-time_samples[0])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        np.abs(coefficients),
        extent=[time_samples.min(), time_samples.max(), scales[-1], scales[0]],
        aspect='auto',
        cmap='viridis'
    )
    ax.set_title("Continuous Wavelet Transform (CWT)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Scale")
    plt.colorbar(im, ax=ax, label="Magnitude")
    st.pyplot(fig)

# Seismic Inversion Feasibility Function
def seismic_inversion_feasibility(logs, wavelet_freq):
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
        st.error(f"Inversion feasibility analysis failed: {str(e)}")
        return None

# Download link generator
def get_table_download_link(df, filename="results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Main processing function with error handling
@handle_errors
def process_data(uploaded_file, model_choice, include_uncertainty=False, mc_iterations=100, **kwargs):
    # Read and validate data
    if isinstance(uploaded_file, str):
        logs = pd.read_csv(uploaded_file)
    else:
        # Handle both CSV and LAS files
        uploaded_file.seek(0)
        if uploaded_file.name.endswith('.las'):
            las = lasio.read(uploaded_file)
            logs = las.df()
            logs['DEPTH'] = logs.index
        else:
            logs = pd.read_csv(uploaded_file)
    
    required_columns = {'DEPTH', 'VP', 'VS', 'RHO', 'VSH', 'SW', 'PHI'}
    if not required_columns.issubset(logs.columns):
        missing = required_columns - set(logs.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Get saturations from kwargs
    sw = kwargs.get('sw', 0.8)
    so = kwargs.get('so', 0.15)
    sg = kwargs.get('sg', 0.05)
    
    # Skip fluid substitution for RPT models (they're visualization-only)
    if model_choice in ["Soft Sand RPT (rockphypy)", "Stiff Sand RPT (rockphypy)"] and rockphypy_available:
        # Just add placeholder columns for consistency
        for case in ['B', 'O', 'G', 'MIX']:
            logs[f'VP_FRM{case}'] = logs.VP
            logs[f'VS_FRM{case}'] = logs.VS
            logs[f'RHO_FRM{case}'] = logs.RHO
            logs[f'IP_FRM{case}'] = logs.VP * logs.RHO
            logs[f'VPVS_FRM{case}'] = logs.VP/logs.VS
            logs[f'LFC_{case}'] = 0  # Default to undefined
        
        return logs, None  # No MC results for RPT models
    
    # Extract parameters from kwargs with defaults
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
    
    # VRH function
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

    # Process data
    shale = logs.VSH.values
    sand = 1 - shale - logs.PHI.values
    shaleN = shale/(shale+sand)
    sandN = sand/(shale+sand)
    k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])

    # Fluid mixtures (using slider values)
    water = sw
    oil = so
    gas = sg
    
    # Effective fluid properties (weighted average)
    rho_fl = water*rho_b + oil*rho_o + gas*rho_g
    k_fl = 1.0 / (water/k_b + oil/k_o + gas/k_g)  # Reuss average for fluid mixture
    
    # Select model function and prepare arguments
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

    # Apply selected model with all required parameters
    if model_choice == "Gassmann's Fluid Substitution":
        # Brine case (100% brine)
        vpb, vsb, rhob, kb = model_func(rho_b, k_b, rho_b, k_b, k0, mu0, logs.PHI)
        
        # Oil case (100% oil)
        vpo, vso, rhoo, ko = model_func(rho_b, k_b, rho_o, k_o, k0, mu0, logs.PHI)
        
        # Gas case (100% gas)
        vpg, vsg, rhog, kg = model_func(rho_b, k_b, rho_g, k_g, k0, mu0, logs.PHI)
        
        # Mixed case (using slider saturations)
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
        vp_mix, vs_mix, rho_mix, _ = model_func(rho_b, k_b, rho_fl, k_fl, k0, mu0, logs.PHI)

    # Litho-fluid classification
    brine_sand = ((logs.VSH <= sand_cutoff) & (logs.SW >= 0.65))
    oil_sand = ((logs.VSH <= sand_cutoff) & (logs.SW < 0.65) & (logs.SW >= 0.35))
    gas_sand = ((logs.VSH <= sand_cutoff) & (logs.SW < 0.35))
    shale_flag = (logs.VSH > sand_cutoff)

    # Add results to logs
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

    # LFC flags (1=Brine, 2=Oil, 3=Gas, 4=Mixed, 5=Shale)
    for case, val in [('B', 1), ('O', 2), ('G', 3), ('MIX', 4)]:
        temp_lfc = np.zeros(np.shape(logs.VSH))
        temp_lfc[brine_sand.values | oil_sand.values | gas_sand.values] = val
        temp_lfc[shale_flag.values] = 5  # Shale
        logs[f'LFC_{case}'] = temp_lfc

    # Uncertainty analysis if enabled
    mc_results = None
    if include_uncertainty:
        # Define parameter distributions
        params = {
            'rho_f1': (rho_b, rho_b_std),
            'k_f1': (k_b, k_b_std),
            'rho_f2': (rho_fl, np.sqrt((sw*rho_b_std)**2 + (so*rho_o_std)**2 + (sg*rho_g_std)**2)),
            'k_f2': (k_fl, np.sqrt((sw*k_b_std)**2 + (so*k_o_std)**2 + (sg*k_g_std)**2)),
            'k0': (k0.mean(), 0.1 * k0.mean()),  # 10% uncertainty in mineral moduli
            'mu0': (mu0.mean(), 0.1 * mu0.mean()),
            'phi': (logs.PHI.mean(), 0.05)  # 5% porosity uncertainty
        }
        
        # Add model-specific parameters
        if model_choice == "Critical Porosity Model (Nur)":
            params['phi_c'] = (critical_porosity, 0.01)
        elif model_choice in ["Contact Theory (Hertz-Mindlin)", "Dvorkin-Nur Soft Sand Model"]:
            params['Cn'] = (coordination_number, 1)
            params['P'] = (effective_pressure, 5)
            if model_choice == "Dvorkin-Nur Soft Sand Model":
                params['phi_c'] = (critical_porosity, 0.01)
        
        # Run Monte Carlo simulation
        mc_results = monte_carlo_simulation(logs, model_func, params, mc_iterations)

    return logs, mc_results

# Main content area
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
            effective_pressure=effective_pressure if 'effective_pressure' in locals() else None
        )
        
        # Depth range selection
        st.header("Well Log Visualization")
        ztop, zbot = st.slider(
            "Select Depth Range", 
            float(logs.DEPTH.min()), 
            float(logs.DEPTH.max()), 
            (float(logs.DEPTH.min()), float(logs.DEPTH.max()))
        )
        
        # Visualization
        ccc = ['#B3B3B3','blue','green','red','magenta','#996633']  # Added magenta for mixed case
        cmap_facies = colors.ListedColormap(ccc[0:len(ccc)], 'indexed')

        # Create a filtered dataframe for the selected depth range
        ll = logs.loc[(logs.DEPTH>=ztop) & (logs.DEPTH<=zbot)]
        cluster = np.repeat(np.expand_dims(ll['LFC_B'].values,1), 100, 1)

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

        # Wedge Model and Frequency Analysis
        st.header("Wedge Model & Frequency Analysis")
        if st.checkbox("Show Wedge Model", value=True):
            synthetic_seismic = plot_wedge_model(
                vp_log=logs.VP.values,
                vs_log=logs.VS.values,
                rho_log=logs.RHO.values,
                depth_log=logs.DEPTH.values,
                wavelet_freq=wavelet_freq,
                thickness_range=(wedge_min, wedge_max)
            )
            
            # Time samples for frequency analysis
            t_samp = np.arange(0, 0.5, 0.0001)
            
            # Frequency analysis
            if st.checkbox("Show Frequency Spectrum & STFT", value=True):
                plot_frequency_analysis(
                    synthetic_seismic,
                    t_samp,
                    np.arange(min_angle, max_angle + 1, angle_step),
                    wavelet_freq
                )
            
            # CWT analysis
            if st.checkbox("Show Continuous Wavelet Transform (CWT)", value=True):
                plot_cwt_analysis(
                    synthetic_seismic,
                    t_samp,
                    np.arange(min_angle, max_angle + 1, angle_step),
                    scales=np.arange(cwt_scales[0], cwt_scales[1] + 1),
                    wavelet=cwt_wavelet
                )

        # [Rest of your existing code for AVO modeling, crossplots, etc...]

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
