# Author: Ryo Segawa (whizznihil.kid@gmail.com)
# Lookup table generation for NME (Neuronal Mechanics Excitation)

"""
Usage:
    python run_lookups.py -fiber_length <length> -fiber_diameter <diameter> \
                         -membrane_thickness <thickness> -freq <frequency> \
                         -amp <amplitude> -charge <charge>

Parameters (all in SI units):
    -fiber_length: Axon fiber length (m), recommended ≥ 3mm
    -fiber_diameter: Axon fiber diameter (m)
    -membrane_thickness: Membrane thickness (m)
    -freq: Ultrasound frequency (Hz)
    -amp: Ultrasound amplitude (Pa)
    -charge: Membrane charge density (nC/cm²)

Optional flags:
    --debug-plot: Generate debug plots showing mechanical deformation
    -v/--verbose: Increase output verbosity

Example:
    python run_lookups.py -fiber_length 1e-3 -fiber_diameter 1e-6 \
                         -membrane_thickness 1.4e-9 -freq 1e6 \
                         -amp 1e6 -charge -65.0
"""

import os
import itertools
import logging
import numpy as np
import itertools # Added for itertools.chain.from_iterable
from neuron import h

from NME.utils import logger, isIterable, LOOKUP_DIR
from NME.core import Lookup, AcousticDrive
from NME.parsers import MechSimParser
from NME.neurons import SundtSegment
from MorphoSONIC.MorphoSONIC.models import UnmyelinatedFiber
from datetime import datetime

def make_cycle_points(f_hz, n_points, phase_rad=0.0):
    """Generate time points for one cycle of frequency f_hz
    Returns:
        t_points: array of time points
        n_points: number of points (same as input for f>0, 1 for f=0)
    """
    if f_hz > 0:
        T = 1.0 / f_hz  # Period in seconds
        return np.linspace(0, T, n_points, endpoint=False), n_points
    else:
        return np.array([0.0]), 1  # Single point for DC/static case

# --- Physical Constants ---
DENSITY_TISSUE = 1000.0  # kg/m^3
SOUND_VELOCITY_MEDIUM = 1500.0  # m/s # Speed of sound in tissue

# --- Helper Functions for Dynamic Membrane Capacitance Calculation ---

def _calculate_instantaneous_displacements(
        P_amplitude_Pa, time_in_cycle_s, omega_rad_per_s, k_rad_per_m,
        initial_length_seg_m, initial_outer_diameter_seg_m,
        density_kg_per_m3, sound_velocity_m_per_s):
    """
    Calculates instantaneous length and radial displacements of an axon segment.
    Implementation of:
    Δl_i(t) = (P_0/ρcω)[cos(kil_i(0) - ωT) - cos(k(i-1)l_i(0) - ωT)]
    Δd_i(t) = D_i(0)/2 * (sqrt(l_i(0)/(l_i(0) + Δl_i(t))) - 1)
    
    Using i=1, l_i(0)=initial_length_seg_m.
    T is time_in_cycle_s (current time relative to ultrasound start).
    """
    l_i_0 = initial_length_seg_m
    coeff = P_amplitude_Pa / (density_kg_per_m3 * sound_velocity_m_per_s * omega_rad_per_s)
    
    # Time thresholds (i=1 case)
    t1 = 0.0  # k(i-1)l_i(0)/ω = 0 for i=1
    t2 = k_rad_per_m * initial_length_seg_m / omega_rad_per_s  # kil_i(0)/ω
    
    # Piecewise calculation
    if time_in_cycle_s <= t1:
        # Case 1: T ≤ 0
        delta_l_inst = 0.0
    elif t1 < time_in_cycle_s <= t2:
        # Case 2: 0 < T ≤ kl_i(0)/ω
        delta_l_inst = coeff * (-np.cos(-omega_rad_per_s * time_in_cycle_s) + 1.0)
    else:
        # Case 3: T > kl_i(0)/ω
        delta_l_inst = coeff * (-np.cos(-omega_rad_per_s * time_in_cycle_s) +
                               np.cos(k_rad_per_m * initial_length_seg_m - omega_rad_per_s * time_in_cycle_s))

    # Check if new length would be negative
    new_length = initial_length_seg_m + delta_l_inst
    if isinstance(new_length, np.ndarray):
        if np.any(new_length <= 0):
            bad_idx = np.where(new_length <= 0)[0]
            for idx in bad_idx:
                logger.error(f"At index {idx} - New length would be negative: {new_length[idx]:.2e} m "
                           f"(initial={initial_length_seg_m:.2e} m, delta={delta_l_inst[idx]:.2e} m)")
            raise ValueError("New length would be negative - physics breakdown")
    else:  # Scalar value
        if new_length <= 0:
            logger.error(f"New length would be negative: {new_length:.2e} m "
                        f"(initial={initial_length_seg_m:.2e} m, delta={delta_l_inst:.2e} m)")
            raise ValueError("New length would be negative - physics breakdown")

    # Calculate radial displacement based on length change
    ratio = l_i_0 / (l_i_0 + delta_l_inst)
    delta_d_radial_inst = initial_outer_diameter_seg_m * (np.sqrt(ratio) - 1.0) / 2.0    # Check if new membrane radius would be negative
    new_radius = initial_outer_diameter_seg_m/2 + delta_d_radial_inst
    if isinstance(new_radius, np.ndarray):
        if np.any(new_radius <= 0):
            bad_idx = np.where(new_radius <= 0)[0]
            for idx in bad_idx:
                logger.error(f"At index {idx} - New membrane radius would be negative: {new_radius[idx]:.2e} m "
                           f"(initial={initial_outer_diameter_seg_m/2:.2e} m, delta={delta_d_radial_inst[idx]:.2e} m)")
            raise ValueError("New membrane radius would be negative - physics breakdown")
    else:  # Scalar value
        if new_radius <= 0:
            logger.error(f"New membrane radius would be negative: {new_radius:.2e} m "
                        f"(initial={initial_outer_diameter_seg_m/2:.2e} m, delta={delta_d_radial_inst:.2e} m)")
            raise ValueError("New membrane radius would be negative - physics breakdown")

    return delta_l_inst, delta_d_radial_inst

def create_debug_plot(t_points, delta_l_values, delta_d_values, cm_values, rm_values, D0_m, d0_m, l0_m, f_hz, A_pa, cm0, rm0,
                     qm_values=None, vm_values=None, fiber_model=None):
    """
    Creates and saves debug plots for displacement values, membrane capacitance, charges, and voltage.
    
    Args:
        t_points: List of time points (s)
        delta_l_values: List of length displacements (m)
        delta_d_values: List of radial displacements (m)
        cm_values: List of membrane capacitance values (µF/cm²)
        D0_m: Initial fiber outer diameter (m)
        d0_m: Initial membrane wall thickness (m)
        l0_m: Initial segment length (m)
        f_hz: Ultrasound frequency (Hz)
        A_pa: Ultrasound amplitude (Pa)
        cm0: Initial membrane capacitance (µF/cm²)
        qm_values: List of membrane charge density values (C/m²)
        vm_values: List of membrane voltage values (mV)
    """
    import matplotlib.pyplot as plt
    
    # Create the figure with appropriate number of subplots
    if qm_values is not None and vm_values is not None:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 18), sharex=True)
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    
    # Plot delta_l
    ax1.plot(np.array(t_points)*1e6, np.array(delta_l_values)*1e9, 'b-', linewidth=2)
    ax1.set_title(f'D={D0_m*1e6:.1f}µm, d={d0_m*1e9:.1f}nm, l0={l0_m*1e6:.1f}µm, f={f_hz*1e-3:.1f}kHz, A={A_pa*1e-3:.1f}kPa')
    ax1.set_ylabel('Δl (nm)')
    ax1.grid(True)
    
    # Plot delta_d
    ax2.plot(np.array(t_points)*1e6, np.array(delta_d_values)*1e9, 'r-', linewidth=2)
    ax2.set_ylabel('Δd (nm)')
    ax2.grid(True)
    
    # Plot Cm (already in µF/cm²)
    ax3.plot(np.array(t_points)*1e6, np.array(cm_values), 'g-', linewidth=2)
    ax3.axhline(y=cm0, color='k', linestyle='--', label='Cm₀')
    ax3.set_ylabel('Cm (µF/cm²)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot Rm (in Ω·cm²)
    ax4.plot(np.array(t_points)*1e6, np.array(rm_values), 'r-', linewidth=2)
    ax4.axhline(y=rm0, color='k', linestyle='--', label='Rm₀')
    ax4.set_ylabel('Rm (Ω·cm²)')
    ax4.legend()
    ax4.grid(True)
    
    # Add Qm and Vm plots if data is provided
    if qm_values is not None and vm_values is not None:
        # Plot Qm (charge density in nC/cm²) - qm_values is already in nC/cm²
        ax5.plot(np.array(t_points)*1e6, np.array(qm_values), 'm-', linewidth=2)
        ax5.set_ylabel('Qm (nC/cm²)')
        ax5.grid(True)
        
        # Plot Vm (membrane voltage in mV)
        ax6.plot(np.array(t_points)*1e6, np.array(vm_values), 'c-', linewidth=2)
        ax6.set_xlabel('Time (µs)')
        ax6.set_ylabel('Vm (mV)')
        ax6.grid(True)
    else:
        # If no Qm/Vm data, put xlabel on Rm plot (ax4)
        ax4.set_xlabel('Time (µs)')
    
    plt.tight_layout()
    
    # Save the figure to a file in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_plots_dir = os.path.join(script_dir, 'debug_plots')
    os.makedirs(debug_plots_dir, exist_ok=True)
    
    plot_filename = os.path.join(debug_plots_dir,
        f'displacements_D{D0_m*1e6:.1f}um_d0{d0_m*1e9:.1f}nm_l0{l0_m*1e6:.1f}um_' +
        f'f{f_hz*1e-3:.1f}kHz_A{A_pa*1e-3:.1f}kPa.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    # Create additional plot for gating variables if fiber_model is UnmyelinatedFiber
    if isinstance(fiber_model, UnmyelinatedFiber) and vm_values is not None:
        # Create SundtSegment for calculating gating variables
        sundt = SundtSegment()
        sundt.celsius = 35.0  # Temperature in ModelDB file (Celsius)
        
        # Calculate gating variables for each voltage point
        states = sundt.steadyStates()
        state_values = {
            name: [func(vm) for vm in vm_values]
            for name, func in states.items()
        }
        
        # Create new figure for gating variables
        fig_gates, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot activation gates (m, n)
        ax1.plot(np.array(t_points)*1e6, state_values['m'], 'b-', label='m (Na+ activation)', linewidth=2)
        ax1.plot(np.array(t_points)*1e6, state_values['n'], 'r-', label='n (K+ activation)', linewidth=2)
        ax1.set_title(f'Gating Variables over Time (Mean Vm={np.mean(vm_values):.1f}mV, f={f_hz*1e-3:.1f}kHz)')
        ax1.set_ylabel('Activation')
        ax1.grid(True)
        ax1.legend()
        
        # Plot inactivation gates (h, l)
        ax2.plot(np.array(t_points)*1e6, state_values['h'], 'g-', label='h (Na+ inactivation)', linewidth=2)
        ax2.plot(np.array(t_points)*1e6, state_values['l'], 'm-', label='l (K+ inactivation)', linewidth=2)
        ax2.set_xlabel('Time (µs)')
        ax2.set_ylabel('Inactivation')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save gating variables plot
        gates_filename = plot_filename.replace('.png', '_gates.png')
        plt.savefig(gates_filename, dpi=300)
        plt.close()
        
        # Calculate gate velocities
        gate_velocities = sundt.derStates()
        velocity_values = {
            name: [func(vm, {k: state_values[k][i] for k in state_values})
                  for i, vm in enumerate(vm_values)]
            for name, func in gate_velocities.items()
        }
        
        # Calculate alpha/beta rates
        alpha_values = {
            f'alpha_{gate}': [getattr(sundt, f'alpha{gate}')(vm) for vm in vm_values]
            for gate in ['m', 'h', 'n', 'l']
        }
        beta_values = {
            f'beta_{gate}': [getattr(sundt, f'beta{gate}')(vm) for vm in vm_values]
            for gate in ['m', 'h', 'n', 'l']
        }
        
        # Gate Velocities Figure
        fig_vel, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot activation velocities
        ax1.plot(np.array(t_points)*1e6, velocity_values['m'], 'b-', label='dm/dt (Na+ activation)', linewidth=2)
        ax1.plot(np.array(t_points)*1e6, velocity_values['n'], 'r-', label='dn/dt (K+ activation)', linewidth=2)
        ax1.set_title('Gate Velocities over Time')
        ax1.set_ylabel('Rate (s^-1)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot inactivation velocities
        ax2.plot(np.array(t_points)*1e6, velocity_values['h'], 'g-', label='dh/dt (Na+ inactivation)', linewidth=2)
        ax2.plot(np.array(t_points)*1e6, velocity_values['l'], 'm-', label='dl/dt (K+ inactivation)', linewidth=2)
        ax2.set_xlabel('Time (µs)')
        ax2.set_ylabel('Rate (s^-1)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        vel_filename = plot_filename.replace('.png', '_velocities.png')
        plt.savefig(vel_filename, dpi=300)
        plt.close()
        
        # Alpha/Beta Rates Figure
        fig_rates, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        t_us = np.array(t_points)*1e6  # Convert to microseconds
        
        # Na+ activation (m) rates
        ax1.plot(t_us, alpha_values['alpha_m'], 'b-', label='α_m', linewidth=2)
        ax1.plot(t_us, beta_values['beta_m'], 'b--', label='β_m', linewidth=2)
        ax1.set_title('Na+ Activation Rates')
        ax1.set_ylabel('Rate (s^-1)')
        ax1.grid(True)
        ax1.legend()
        
        # Na+ inactivation (h) rates
        ax2.plot(t_us, alpha_values['alpha_h'], 'g-', label='α_h', linewidth=2)
        ax2.plot(t_us, beta_values['beta_h'], 'g--', label='β_h', linewidth=2)
        ax2.set_title('Na+ Inactivation Rates')
        ax2.grid(True)
        ax2.legend()
        
        # K+ activation (n) rates
        ax3.plot(t_us, alpha_values['alpha_n'], 'r-', label='α_n', linewidth=2)
        ax3.plot(t_us, beta_values['beta_n'], 'r--', label='β_n', linewidth=2)
        ax3.set_title('K+ Activation Rates')
        ax3.set_xlabel('Time (µs)')
        ax3.set_ylabel('Rate (s^-1)')
        ax3.grid(True)
        ax3.legend()
        
        # K+ inactivation (l) rates
        ax4.plot(t_us, alpha_values['alpha_l'], 'm-', label='α_l', linewidth=2)
        ax4.plot(t_us, beta_values['beta_l'], 'm--', label='β_l', linewidth=2)
        ax4.set_title('K+ Inactivation Rates')
        ax4.set_xlabel('Time (µs)')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        rates_filename = plot_filename.replace('.png', '_rates.png')
        plt.savefig(rates_filename, dpi=300)
        plt.close()
        
        logger.info(f"Debug plots saved:")
        logger.info(f"  • Main plot: {plot_filename}")
        logger.info(f"  • Gating variables: {gates_filename}")
        logger.info(f"  • Gate velocities: {vel_filename}")
        logger.info(f"  • Rate constants: {rates_filename}")


def _calculate_instantaneous_membrane_resistance(
        Rm0_ohm_cm2,  # Initial membrane resistance (Ω·cm²)
        initial_length_seg_m,  # Initial length (m)
        initial_outer_diameter_seg_m,  # Initial outer diameter (m)
        initial_wall_thickness_seg_m,  # Initial membrane thickness (m)
        delta_l_inst_m,  # Length displacement (m)
        delta_d_radial_inst_m  # Radial displacement (m)
        ):
    """
    Calculates instantaneous membrane resistance based on deformations.
    R_m(t) = R_m(0) * (l(0)D(0))/((l(0)+Δl(t))(D(0)+2Δd(t))) * (d(0)+Δd(t))/d(0)
    """
    # Calculate geometry factors
    length_diam_factor = ((initial_length_seg_m * initial_outer_diameter_seg_m) /
                         ((initial_length_seg_m + delta_l_inst_m) *
                          (initial_outer_diameter_seg_m + 2*delta_d_radial_inst_m)))
    
    thickness_factor = (initial_wall_thickness_seg_m + delta_d_radial_inst_m) / initial_wall_thickness_seg_m
    
    # Calculate new resistance
    Rm_inst = Rm0_ohm_cm2 * length_diam_factor * thickness_factor
    
    return Rm_inst

def _calculate_instantaneous_specific_cm(
        cm0_uF_per_cm2,
        initial_outer_diameter_seg_m, initial_wall_thickness_seg_m, initial_length_seg_m,
        delta_l_inst_m, delta_d_radial_inst_m):
    """
    Calculates instantaneous specific Cm based on deformations.
    Implementation of:
    Cm,i(t) = Cm,i(0) * [ln(D_i(0)/(D_i(0)-2d_i(t)))] / [ln((D_i(0)+2Δd_i(t))/(D_i(0)-2d_i(t)))] * (l_i(0)+Δl_i(t))/l_i(0)
    
    Args:
        cm0_uF_per_cm2: Initial specific membrane capacitance in µF/cm²
    Returns:
        Instantaneous specific membrane capacitance in µF/cm²
    """
    # Basic safety checks
    if initial_wall_thickness_seg_m <= 0:
        logger.debug(f"Wall thickness non-positive: {initial_wall_thickness_seg_m:.2e} m. Using resting Cm.")
        return cm0_uF_per_cm2

    # Calculate the denominator term D_i(0)-2d_i(t)
    denominator = initial_outer_diameter_seg_m - 2 * initial_wall_thickness_seg_m

    # Calculate D_i(0)+2Δd_i(t)
    deformed_outer_diameter = initial_outer_diameter_seg_m + 2 * delta_d_radial_inst_m

    # Calculate logarithmic terms
    numerator_log = np.log(initial_outer_diameter_seg_m / denominator)
    denominator_log = np.log(deformed_outer_diameter / denominator)
        
    membrane_factor = numerator_log / denominator_log

    # Calculate length factor (l_i(0)+Δl_i(t))/l_i(0)
    current_length = initial_length_seg_m + delta_l_inst_m
    length_factor = current_length / initial_length_seg_m

    # Calculate final capacitance
    cm_inst = cm0_uF_per_cm2 * membrane_factor * length_factor
    return cm_inst

# --- End of helper functions ---

def computeAStimLookup(fiber_model, D0ref, d0ref, L0ref, l0ref, fref, Aref, Qref,
                        n_points_per_cycle=1000, debug_plot=False, loglevel=logging.INFO):
    ''' Run calculations for lookup table with geometric axes:
        - D0: fiber outer diameter (m)
        - d0: membrane wall thickness (m)
        
        UnmyelinatedFiber parameters:
        - D0 axis (from --D0_fiber_diameter) defines fiber outer diameters
        - d0 axis (from --d0_membrane_thickness) defines membrane wall thicknesses
        - Dynamic Cm calculated from current D0, d0 values and deformation

        Other axes: characteristic lengths (l0), frequencies (f),
        amplitudes (A), and charge densities (Q).

        :param fiber_model: UnmyelinatedFiber instance.
        :param D0ref: Array of fiber diameters (m).
        :param d0ref: Array of membrane wall thicknesses (m).
        :param l0ref: Array of characteristic lengths (m).
        :param fref: Array of acoustic frequencies (Hz).
        :param Aref: Array of acoustic amplitudes (Pa).
        :param Qref: Array of membrane charge densities (C/m2).
        :param loglevel: Logging level.
        :return: Lookup object with n-dimensional arrays.
    '''
    logger.setLevel(loglevel)
    
    # Verify parameters
    if not isinstance(fiber_model, UnmyelinatedFiber):
        raise ValueError("fiber_model must be an UnmyelinatedFiber instance")

    param_descs = {
        'D0': 'Fiber diameter (m)',
        'd0': 'Membrane thickness (m)',
        'L0': 'Fiber length (m)',
        'l0': 'Section length (m)',
        'f': 'Ultrasound frequency (Hz)',
        'A': 'Ultrasound amplitude (Pa)',
        'Q': 'Charge density (nC/cm²)'
    }

    # Populate reference vectors dictionary
    refs = {
        'D0': D0ref, # m, fiber outer diameter
        'd0': d0ref, # m, membrane wall thickness
        'L0': L0ref, # m, fiber length
        'f': fref,   # Hz
        'A': Aref,   # Pa
        'Q': Qref    # C/m2
    }

    # l0ref will be used for section length calculations but not included in refs

    # Validate input arrays
    for key, values in refs.items():
        if not isinstance(values, np.ndarray):
            raise TypeError(f'{param_descs[key]} must be provided as numpy array')
        if values.size == 0:
            raise ValueError(f'{param_descs[key]} array cannot be empty')
        if key in ('D0', 'd0', 'l0', 'f') and np.any(values <= 0):
            raise ValueError(f'{param_descs[key]} values must be positive')
        if key == 'A' and np.any(values < 0):
            raise ValueError(f'{param_descs[key]} values must be non-negative')
            
    # Create simulation queue
    # The queue will iterate over combinations of parameters.
    
    param_combinations = []
    # Base parameters (D0, d0, L0, f, A, Q)
    base_param_list = list(itertools.product(refs['D0'], refs['d0'], refs['L0'], refs['f'], refs['A'], refs['Q']))
    
    for base_params in base_param_list:
        param_combinations.append(base_params)

    # Get references dimensions for the final lookup table
    dims = np.array([x.size for x in refs.values()]) 
    
    logger.info(f"Lookup dimensions: {dims}, Keys: {list(refs.keys())}")
    outputs_effvars = []
    outputs_tcomps = [] # Placeholder for computation times

    # Get capacitance value from first node section (in uF/cm^2)
    section = fiber_model.nodes['node0']  # Access the first node section
    Cm_axon = section.cm  # Already in uF/cm^2
    logger.info(f"Using fiber capacitance: {Cm_axon:.2f} µF/cm²")
    for D0_val, d0_val, L0_val, f_val, A_val, Q_val in param_combinations:
        # Both D0 (fiber diameter/effective thickness) and d0 (membrane thickness) values are available
        # l0_val is used in UMF dynamic Cm calculations
        drive = AcousticDrive(f_val, A_val) # Create drive for current f, A

        # Generate time points for this cycle
        t_points, cycle_points = make_cycle_points(f_val, n_points_per_cycle)
        
        # Use a constant charge value for the cycle
        Qm_cycle = np.full(cycle_points, Q_val)

        # Determine capacitance for the cycle based on US amplitude A_val
        if A_val == 0:  # No US, use neuron's resting capacitance (already in uF/cm^2)
            Cm_cycle_array = np.full(cycle_points, Cm_axon)
        else:  # US is ON (A_val > 0)
            # --- Dynamic Cm calculation for UnmyelinatedFiber ---
                current_initial_outer_diameter_seg_m = D0_val
                current_initial_membrane_thickness_seg_m = d0_val
                current_initial_length_one_segment_m = l0ref[0]  # Use the section length from l0ref

                k_wave_number = (2 * np.pi * f_val) / SOUND_VELOCITY_MEDIUM
                t_cycle, n_points = make_cycle_points(f_val, n_points_per_cycle)
                
                # Pre-allocate arrays for the cycle
                cm_cycle_list = np.zeros(n_points)
                rm_cycle_list = np.zeros(n_points)
                
                # Arrays for plotting
                t_points_plot = []
                delta_l_values = []
                delta_d_values = []
                
                # Calculate initial membrane resistance based on tutorial's approach
                # R_m,i(0) = ρ_m / (π * l_i(0) * D_i(0))
                # where ρ_m is membrane resistivity in Ω·cm² from membrane conductance (S/cm²)
                # Get specific membrane conductance from one section
                section = fiber_model.nodes['node0']  # Get the first node section
                
                # Get passive conductance from mechanisms or use default value Ω·cm²
                gLeak = SundtSegment.gLeak  # S/m² (non-specific leakage)
                rho_m = 1.0 / gLeak  # Convert to resistivity Ω·m²
                Rm0 = (rho_m*1e4) / (np.pi * (current_initial_length_one_segment_m *1e2) * (current_initial_outer_diameter_seg_m * 1e2))

                # Calculate displacements for all time points
                for t_idx, t_inst_s in enumerate(t_cycle):
                    # Calculate displacements
                    delta_l, delta_d_rad = _calculate_instantaneous_displacements(
                        P_amplitude_Pa=A_val, time_in_cycle_s=t_inst_s,
                        omega_rad_per_s=(2 * np.pi * f_val), k_rad_per_m=k_wave_number,
                        initial_length_seg_m=current_initial_length_one_segment_m,
                        initial_outer_diameter_seg_m=current_initial_outer_diameter_seg_m,
                        density_kg_per_m3=DENSITY_TISSUE,
                        sound_velocity_m_per_s=SOUND_VELOCITY_MEDIUM
                    )
                    
                    # Store for plotting
                    t_points_plot.append(t_inst_s)
                    delta_l_values.append(delta_l)
                    delta_d_values.append(delta_d_rad)

                    # Calculate capacitance and membrane resistance
                    cm_inst = _calculate_instantaneous_specific_cm(
                        cm0_uF_per_cm2=Cm_axon,
                        initial_outer_diameter_seg_m=current_initial_outer_diameter_seg_m,
                        initial_wall_thickness_seg_m=current_initial_membrane_thickness_seg_m,
                        initial_length_seg_m=current_initial_length_one_segment_m,
                        delta_l_inst_m=delta_l,
                        delta_d_radial_inst_m=delta_d_rad
                    )
                    cm_cycle_list[t_idx] = cm_inst

                    # Calculate membrane resistance
                    rm_inst = _calculate_instantaneous_membrane_resistance(
                        Rm0_ohm_cm2=Rm0,
                        initial_length_seg_m=current_initial_length_one_segment_m,
                        initial_outer_diameter_seg_m=current_initial_outer_diameter_seg_m,
                        initial_wall_thickness_seg_m=current_initial_membrane_thickness_seg_m,
                        delta_l_inst_m=delta_l,
                        delta_d_radial_inst_m=delta_d_rad
                    )
                    rm_cycle_list[t_idx] = rm_inst

                # Convert lists to numpy arrays
                Cm_cycle_array = np.array(cm_cycle_list)
                Rm_cycle_array = np.array(rm_cycle_list)
            

            
        # Calculate membrane voltage using:
        # - Qm: membrane charge density (input in nC/cm²)
        # - Cm: membrane capacitance (in µF/cm²)
        # The units work out: (nC/cm²) / (µF/cm²) = mV
        Vm_cycle = Qm_cycle / Cm_cycle_array  # Result in mV
                
        # Calculate mean membrane potential and resistance
        mean_vm = np.mean(Vm_cycle)
        mean_rm = np.mean(Rm_cycle_array)
        effvars_dict = {
            'V': mean_vm,      # Mean membrane potential (mV)
            'Rm': mean_rm      # Mean membrane resistance (Ω·cm²)
        }

        # If model is UnmyelinatedFiber, use Sundt model for ionic mechanisms
        if isinstance(fiber_model, UnmyelinatedFiber):
            # Create SundtSegment for ionic mechanisms calculations
            sundt = SundtSegment()
            sundt.celsius = 35.0  # Temperature in ModelDB file (Celsius)
            
            # Calculate gating variables
            states = sundt.steadyStates()
            state_values = {name: func(mean_vm) for name, func in states.items()}
            
            # Calculate ionic currents
            currents = sundt.currents()
            current_values = {name: func(mean_vm, state_values) for name, func in currents.items()}
            
            # Calculate gate velocities
            gate_velocities = sundt.derStates()
            velocity_values = {
                f'{name}_velocity': func(mean_vm, state_values)
                for name, func in gate_velocities.items()
            }
            
            # Calculate alpha and beta rates for each gate
            alpha_values = {
                f'alpha_{gate}': getattr(sundt, f'alpha{gate}')(mean_vm)
                for gate in ['m', 'h', 'n', 'l']
            }
            
            beta_values = {
                f'beta_{gate}': getattr(sundt, f'beta{gate}')(mean_vm)
                for gate in ['m', 'h', 'n', 'l']
            }
            
            # Add all variables to effvars dictionary
            effvars_dict.update(state_values)      # Gating variables (m, h, n, l)
            effvars_dict.update(current_values)    # Ionic currents (iNa, iKd, iLeak)
            effvars_dict.update(velocity_values)   # Gate velocities (m_velocity, etc.)
            effvars_dict.update(alpha_values)      # Alpha rates (alpha_m, etc.)
            effvars_dict.update(beta_values)       # Beta rates (beta_m, etc.)
            
            # Log all calculated variables
            logger.info("\nSundt model variables calculated:")
            logger.info("  • Membrane potential: {:.2f} mV".format(mean_vm))
            logger.info("  • Gating variables:")
            for gate in ['m', 'h', 'n', 'l']:
                logger.info(f"    - {gate}: {state_values[gate]:.3f}")
                logger.info(f"      velocity: {velocity_values[f'{gate}_velocity']:.3f} s^-1")
                logger.info(f"      alpha: {alpha_values[f'alpha_{gate}']:.3f} s^-1")
                logger.info(f"      beta: {beta_values[f'beta_{gate}']:.3f} s^-1")
            logger.info("  • Ionic currents:")
            for curr, val in current_values.items():
                logger.info(f"    - {curr}: {val:.2f} mA/m²")
        else:
            logger.warning(f"No ionic mechanism model defined for {type(fiber_model).__name__}")
            
        outputs_effvars.append(effvars_dict) # Append the single dict for this combination
        
        outputs_tcomps.append(0.0) # One tcomp entry per main param combination

        # Create debug plot if enabled (after all calculations)
        if debug_plot:  # Only for ultrasound cases
            create_debug_plot(
                t_points_plot, delta_l_values, delta_d_values, cm_cycle_list, rm_cycle_list,
                current_initial_outer_diameter_seg_m,
                current_initial_membrane_thickness_seg_m,
                current_initial_length_one_segment_m,
                f_val, A_val, Cm_axon, Rm0,
                qm_values=Qm_cycle, vm_values=Vm_cycle,
                fiber_model=fiber_model
            )

    nout = len(outputs_effvars)
    ncombs = dims.prod()
    if nout != ncombs:
        raise ValueError(
            f'Number of outputs ({nout}) does not match number of input combinations ({ncombs})')

    # Reshape effective variables (outputs_effvars)
    tables = {}
    # Add section length as a constant table value
    l0_shape = list(dims)
    tables['l0'] = np.full(l0_shape, l0ref[0])  # Use the first value from l0ref
    
    # Add other table values
    for key in outputs_effvars[0].keys():
        effvar_values = [ev[key] for ev in outputs_effvars]
        tables[key] = np.array(effvar_values).reshape(dims)


    # Reshape computation times (outputs_tcomps)
    if not outputs_tcomps: # If param_combinations was empty
        # Ensure dims is not empty before trying to create an empty array with it
        tcomps_final_shape = dims if dims.size > 0 else (0,)
        tcomps_final = np.array([]).reshape(tcomps_final_shape) if np.prod(dims) == 0 else np.zeros(tcomps_final_shape)

    else:
        try:
            tcomps_final = np.array(outputs_tcomps).reshape(dims)
        except ValueError as e:
            logger.error(f"Error reshaping tcomps. Expected shape {dims} (from keys {list(refs.keys())}), "
                         f"got {len(outputs_tcomps)} elements.")
            raise e
            
    tables['tcomp'] = tcomps_final

    # Construct and return lookup object
    return Lookup(refs, tables)


# --- Helper Function for Augmenting UnmyelinatedFiber Instances ---
# --- Main Function ---
def main():

    # Temporary debug arguments - hardcoded for testing
    debug_args = {
        'fiber_length': 1e-3,           # 1 mm
        'fiber_diameter': 1e-6,         # 1 µm  
        'membrane_thickness': 1.4e-9,   # 1.4 nm
        'freq': 1e6,                    # 1 MHz
        'amp': 1e6,                     # 1 MPa
        'charge': -65.0,                # -65 nC/cm²
        'debug_plot': True,             # Enable debug plots
        'verbose': True                 # Enable verbose logging
    }
    
    import argparse
    parser = argparse.ArgumentParser(description='Generate lookup tables for fiber simulations')
    
    # Add command line arguments (keeping for future use)
    parser.add_argument('-fiber_length', type=float, required=False, default=debug_args['fiber_length'], help='Fiber length (m)')
    parser.add_argument('-fiber_diameter', type=float, required=False, default=debug_args['fiber_diameter'], help='Fiber diameter (m)')
    parser.add_argument('-membrane_thickness', type=float, required=False, default=debug_args['membrane_thickness'], help='Membrane thickness (m)')
    parser.add_argument('-freq', type=float, required=False, default=debug_args['freq'], help='Frequency (Hz)')
    parser.add_argument('-amp', type=float, required=False, default=debug_args['amp'], help='Amplitude (Pa)')
    parser.add_argument('-charge', type=float, required=False, default=debug_args['charge'], help='Charge density (C/m²)')
    parser.add_argument('--debug-plot', action='store_true', default=debug_args['debug_plot'], help='Generate debug plots showing mechanical deformation')
    parser.add_argument('-v', '--verbose', action='store_true', default=debug_args['verbose'], help='Increase verbosity')
    args = parser.parse_args()

    # Convert arguments to numpy arrays for compatibility
    args_dict = {
        'fiber_length': np.array([args.fiber_length]),
        'fiber_diameter': np.array([args.fiber_diameter]),
        'membrane_thickness': np.array([args.membrane_thickness]),  # Already in meters
        'freq': np.array([args.freq]),
        'amp': np.array([args.amp]),
        'charge': np.array([args.charge]),
        'loglevel': logging.DEBUG if args.verbose else logging.INFO,
        'debug_plot': args.debug_plot
    }
    
    logger.setLevel(args_dict['loglevel'])

    # Create output directory
    current_date = datetime.now().strftime("%Y%m%d")
    lookup_dir = os.path.join(LOOKUP_DIR, f'unmyelinated_fiber_{current_date}')
    os.makedirs(lookup_dir, exist_ok=True)

    # Create all parameter combinations
    param_combinations = list(itertools.product(
        args_dict['fiber_diameter'],     # Fiber diameters
        args_dict['membrane_thickness'],  # Membrane thicknesses
        args_dict['fiber_length'],       # Fiber lengths
        args_dict['freq'],               # Frequencies
        args_dict['amp'],                # Amplitudes
        args_dict['charge']              # Charge densities
    ))

    # Print configuration summary
    logger.info("\nLookup Table Configuration:")
    logger.info(f"  Total combinations to process: {len(param_combinations)}")
    
    logger.info("\n  Parameter Ranges:")
    logger.info(f"  • Diameters (D0): 1 value: {args_dict['fiber_diameter'][0]*1e6:.1f} µm")
    logger.info(f"  • Membrane Thickness (d0): 1 value: {args_dict['membrane_thickness'][0]*1e9:.1f} nm")
    logger.info(f"  • Fiber Length (L0): 1 value: {args_dict['fiber_length'][0]*1e3:.2f} mm")
    logger.info(f"  • Frequency (f): 1 value: {args_dict['freq'][0]*1e-3:.1f} kHz")
    logger.info(f"  • Amplitude (A): 1 value: {args_dict['amp'][0]*1e-3:.1f} kPa")
    logger.info(f"  • Charge Density (Q): {args_dict['charge'][0]} nC/cm²")

    # Process each combination
    for idx, (fiber_diameter, membrane_thickness, fiber_length, freq, amp, charge) in enumerate(param_combinations):
        logger.info(f"\nProcessing combination {idx + 1}/{len(param_combinations)}:")
        logger.info(f"  • Diameter: {fiber_diameter*1e6:.1f} µm")
        logger.info(f"  • Membrane thickness: {membrane_thickness*1e9:.1f} nm")
        logger.info(f"  • Length: {fiber_length*1e3:.1f} mm")
        logger.info(f"  • Frequency: {freq*1e-3:.1f} kHz")
        logger.info(f"  • Amplitude: {amp*1e-3:.1f} kPa")
        logger.info(f"  • Charge density: {charge:.1f} nC/cm²")

        # Create fiber for this combination
        fiber = UnmyelinatedFiber(fiberD=fiber_diameter, fiberL=fiber_length)
        sec_length = fiber_length / float(fiber.nnodes)

        # Create lookup table for this combination
        lkp = computeAStimLookup(
            fiber,
            D0ref=np.array([fiber_diameter]),       # Single diameter
            d0ref=np.array([membrane_thickness]),   # Single membrane thickness
            L0ref=np.array([fiber_length]),        # Fiber length
            l0ref=np.array([sec_length]),          # Section length
            fref=np.array([freq]),                 # Single frequency
            Aref=np.array([amp]),                  # Single amplitude
            Qref=np.array([charge]),               # Single charge density
            debug_plot=args_dict['debug_plot'],
            loglevel=args_dict['loglevel']
        )

        # Generate unique filename for this combination
        lookup_path = os.path.join(lookup_dir,
            f'lookup_axonD{fiber_diameter*1e6:.1f}um_' +
            f'membraneD{membrane_thickness*1e9:.1f}nm_' +
            f'axonL{fiber_length*1e3:.1f}mm_' +
            f'f{freq*1e-3:.0f}kHz_' +
            f'A{amp*1e-3:.0f}kPa_' +
            f'Q{charge:.1f}nCcm2.pkl')

        lkp.toPickle(lookup_path)
        logger.info(f'  Saved lookup table to: {lookup_path}')


if __name__ == '__main__':
    main()


