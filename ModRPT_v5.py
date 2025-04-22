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
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift

# Import rockphypy with error handling
try:
    from rockphypy import QI, GM, Fluid
    rockphypy_available = True
except ImportError:
    rockphypy_available = False

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced Rock Physics & AVO Modeling with Wedge Modeling")

# ==============================================
# Core Rock Physics Functions
# ==============================================

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

# ==============================================
# AVO and Seismic Functions
# ==============================================

def ricker_wavelet(frequency, length=0.128, dt=0.001, phase=0):
    """Generate a Ricker wavelet with phase rotation"""
    t = np.linspace(-length/2, length/2, int(length/dt))
    y = (1 - 2*(np.pi**2)*(frequency**2)*(t**2)) * np.exp(-(np.pi**2)*(frequency**2)*(t**2))
    
    if phase != 0:
        phase = phase*np.pi/180.0
        yh = signal.hilbert(y)
        yh = np.imag(yh)
        y = np.cos(phase)*y - np.sin(phase)*yh
    
    return t, y

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
        
        # Predictions
        vp_pred = vp_model.predict(X_test_scaled)
        vs_pred = vs_model.predict(X_test_scaled)
        
        # Metrics
        vp_r2 = r2_score(y_vp_test, vp_pred)
        vs_r2 = r2_score(y_vs_test, vs_pred)
        vp_rmse = np.sqrt(mean_squared_error(y_vp_test, vp_pred))
        vs_rmse = np.sqrt(mean_squared_error(y_vs_test, vs_pred))
        
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

def get_table_download_link(df, filename="results.csv"):
    """Generate a download link for CSV files"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def perform_time_frequency_analysis(logs, angles, wavelet_freq, cwt_scales, cwt_wavelet, middle_top, middle_bot, sw, so, sg):
    """Perform comprehensive time-frequency analysis on synthetic gathers"""
    cases = ['Brine', 'Oil', 'Gas', 'Mixed']
    case_data = {
        'Brine': {'vp': 'VP_FRMB', 'vs': 'VS_FRMB', 'rho': 'RHO_FRMB', 'color': 'b'},
        'Oil': {'vp': 'VP_FRMO', 'vs': 'VS_FRMO', 'rho': 'RHO_FRMO', 'color': 'g'},
        'Gas': {'vp': 'VP_FRMG', 'vs': 'VS_FRMG', 'rho': 'RHO_FRMG', 'color': 'r'},
        'Mixed': {'vp': 'VP_FRMMIX', 'vs': 'VS_FRMMIX', 'rho': 'RHO_FRMMIX', 'color': 'm'}
    }

    # Generate synthetic gathers for all cases
    wlt_time, wlt_amp = ricker_wavelet(wavelet_freq)
    t_samp = np.arange(0, 0.5, 0.001)  # Higher resolution for better CWT
    t_middle = 0.2
    
    # Store all gathers for time-frequency analysis
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
            
            rc_series = np.zeros(len(t_samp))
            idx_middle = np.argmin(np.abs(t_samp - t_middle))
            rc_series[idx_middle] = rc
            
            syn_trace = np.convolve(rc_series, wlt_amp, mode='same')
            syn_gather.append(syn_trace)
        
        all_gathers[case] = np.array(syn_gather)
    
    return all_gathers, t_samp

def plot_frequency_analysis(all_gathers, t_samp, angles, wavelet_freq, time_range, freq_range):
    """Plot frequency domain analysis (FFT) with frequency-based color coding"""
    st.subheader("Frequency Domain Analysis (FFT)")
    
    fig_freq, ax_freq = plt.subplots(1, 4, figsize=(24, 5))  # Added column for mixed case
    
    for idx, case in enumerate(all_gathers.keys()):
        syn_gather = all_gathers[case]  # Shape: (num_angles, num_time_samples)
        
        # Time range filtering
        time_mask = (t_samp >= time_range[0]) & (t_samp <= time_range[1])
        t_samp_filtered = t_samp[time_mask]
        
        # Compute FFT parameters
        n = syn_gather.shape[1]  # Number of time samples
        dt = t_samp[1] - t_samp[0]
        freqs = np.fft.rfftfreq(n, dt)  # Frequency bins
        
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
        ax_freq[idx].set_ylim(time_range[1], time_range[0])  # Inverted for seismic display
        ax_freq[idx].set_xlim(freq_range[0], freq_range[1])  # Focus on relevant frequencies
        
        plt.colorbar(im, ax=ax_freq[idx], label='Normalized Amplitude')
    
    plt.tight_layout()
    st.pyplot(fig_freq)

def plot_cwt_analysis(all_gathers, t_samp, angles, cwt_scales, cwt_wavelet, wavelet_freq, time_range, freq_range):
    """Plot CWT analysis with frequency-based color coding and extract 0.20s data"""
    st.subheader("Time-Frequency Analysis (CWT)")
    
    try:
        scales = np.arange(cwt_scales[0], cwt_scales[1]+1)
        # Convert scales to approximate frequencies
        freqs = pywt.scale2frequency(cwt_wavelet, scales) / (t_samp[1]-t_samp[0])
        
        fig_cwt, ax_cwt = plt.subplots(3, len(all_gathers), figsize=(24, 12))  # Adjusted for 4 cases
        
        # Time range filtering
        time_mask = (t_samp >= time_range[0]) & (t_samp <= time_range[1])
        t_samp_filtered = t_samp[time_mask]
        
        # Store CWT magnitudes at t=0.20s for each case (using raw magnitudes)
        cwt_at_020s = {'Case': [], 'Frequency': [], 'Magnitude': []}
        
        for col_idx, case in enumerate(all_gathers.keys()):
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
            ax_cwt[0, col_idx].set_ylim(time_range[1], time_range[0])  # Inverted y-axis
            ax_cwt[0, col_idx].set_xlim(freq_range[0], freq_range[1])
            plt.colorbar(im, ax=ax_cwt[0, col_idx], label='Normalized Magnitude')
            
            # Plot time series at middle angle (inverted y-axis)
            mid_angle_idx = len(angles) // 2
            time_series = syn_gather[mid_angle_idx, time_mask]
            ax_cwt[1, col_idx].plot(t_samp_filtered, time_series, 'k-')
            ax_cwt[1, col_idx].set_title(f"{case} - Time Series (@ {angles[mid_angle_idx]}°)")
            ax_cwt[1, col_idx].set_xlabel("Time (s)")
            ax_cwt[1, col_idx].set_ylabel("Amplitude")
            ax_cwt[1, col_idx].grid(True)
            ax_cwt[1, col_idx].set_xlim(time_range[0], time_range[1])
            ax_cwt[1, col_idx].set_ylim(time_range[1], time_range[0])  # Inverted y-axis
            
            # Plot dominant frequency (inverted y-axis)
            if cwt_magnitudes.size > 0:
                max_freq_indices = np.argmax(cwt_magnitudes, axis=1)
                dominant_freqs = freqs[max_freq_indices]
                
                ax_cwt[2, col_idx].plot(t_samp_filtered, dominant_freqs, 'r-')
                ax_cwt[2, col_idx].set_title(f"{case} - Dominant Frequency")
                ax_cwt[2, col_idx].set_xlabel("Time (s)")
                ax_cwt[2, col_idx].set_ylabel("Frequency (Hz)")
                ax_cwt[2, col_idx].grid(True)
                ax_cwt[2, col_idx].set_ylim(freq_range[1], freq_range[0])  # Inverted y-axis
            
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
            
            for case in all_gathers.keys():
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
            ax_freq_mag.set_xlim(freq_range[0], freq_range[1])
            
            st.pyplot(fig_freq_mag)
        
    except Exception as e:
        st.error(f"Error in CWT analysis: {str(e)}")
