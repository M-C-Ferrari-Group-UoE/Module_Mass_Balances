import math
import warnings
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import least_squares
import numpy as np
import pandas as pd

R_GAS = 8.314  # J/(mol.K)

def mass_balance_CC_ODE_BVP_dP(vars):

    Membrane, Component_properties, Fibre_Dimensions = vars
    Membrane["Total_Flow"] = Membrane["Feed_Flow"] + Membrane["Sweep_Flow"]
    Fibre_Dimensions["Number_Fibre"] = Membrane["Area"] / (Fibre_Dimensions["Length"] * math.pi * Fibre_Dimensions["D_out"])

    epsilon = 1e-10

    J = len(Membrane["Feed_Composition"])

    L = Fibre_Dimensions["Length"]

    Membrane["Feed_Composition"]  = np.array(Membrane["Feed_Composition"])
    Membrane["Sweep_Composition"] = np.array(Membrane["Sweep_Composition"])

    # Free cross-sectional area on the shell side of one module (module area minus fibre area)
    A_shell = (math.pi / 4 * Fibre_Dimensions["D_Module"]**2) - (Fibre_Dimensions["Fibre_per_Module"] * math.pi / 4 * Fibre_Dimensions["D_out"]**2)


    '''----------------------------------------------------------###
    ###------------- Mixture Viscosity Calculation --------------###
    ###----------------------------------------------------------'''

    def mixture_visc(composition): #vectorised for 2D arrays of compositions, returns array of viscosities for each point in the profile
        # Wilke mixing rule:
        # phi_ij = [1 + (mu_i/mu_j)^0.5 * (M_j/M_i)^0.25]^2 / [8 * (1 + M_i/M_j)]^0.5
        y = composition #shape (J, n_points)
        params = np.array(Component_properties["Viscosity_param"])
        visc = 1e-6 * (params[:, 0] * Membrane["Temperature"] + params[:, 1])  # shape (J,) in Pa.s (from trend obtained from NIST)
        Mw = np.array(Component_properties["Molar_mass"])  # shape (J,)
        visc_ratio = np.sqrt(visc[:, None] / visc[None, :])  # (mu_i/mu_j)^0.5, shape (J, J)
        Mw_ratio = (Mw[None, :] / Mw[:, None]) ** 0.25        # (M_j/M_i)^0.25, shape (J, J) 
        phi = ( ( 1 + visc_ratio * Mw_ratio ) **2 ) / ( ( 8 * ( 1 + Mw[:, None]/Mw[None, :]) )**0.5 )  # shape (J, J)

        denominator = (phi @ y).T #matrix multiplication, shape (n_points, J)
        nu = (y.T * visc) / (denominator + epsilon)  # shape (n_points, J)
        visc_mix = np.sum(nu, axis=1)  # shape (n_points,)

        return visc_mix

    '''----------------------------------------------------------###
    ###--------------- Pressure Drop Calculation ----------------###
    ###----------------------------------------------------------'''

    def pressure_drop_permeate(composition, Q, P):
        # Bore side, Hagen-Poiseuille in one fibre: dP/dz = 128 * mu * Qv / (pi * D_in^4)
        # composition shape: (J, n_pts)
        # Q shape: (n_pts,) molar flow in mol/s
        # P shape: (n_pts,) in Pa
        visc_mix = mixture_visc(composition)   # shape (n_pts,)
        D_in = Fibre_Dimensions["D_in"]
        Qv = Q * R_GAS * Membrane["Temperature"] / P      # mol/s -> m3/s at local conditions
        Qv_per_fibre = Qv / Fibre_Dimensions['Number_Fibre']
        dP_dz = (128 * visc_mix * Qv_per_fibre) / (math.pi * D_in**4)  # Pa/m
        return dP_dz   # shape (n_pts,)

    def pressure_drop_retentate(composition, Q, P):
        # Shell side, laminar flow on the hydraulic diameter: dP/dz = 32 * mu * v / Dh^2 with v = Qv / A_shell
        visc_mix = mixture_visc(composition)   # shape (n_pts,)
        D_hyd = Fibre_Dimensions["D_hydraulic"]
        Qv = Q * R_GAS * Membrane["Temperature"] / P      # mol/s -> m3/s at local conditions
        Qv_per_module = Qv / Fibre_Dimensions['Number_Module']
        velocity = Qv_per_module / A_shell                # superficial velocity in the shell, m/s
        dP_dz = (32 * visc_mix * velocity) / D_hyd**2     # Pa/m
        return dP_dz   # shape (n_pts,)

    '''----------------------------------------------------------###
    ###---------- Permeate Composition (closed-end safe) --------###
    ###----------------------------------------------------------'''

    permeance_arr = np.array(Membrane["Permeance"])
    eps_flow      = 1e-6   # normalised permeate flow below which the local flux composition takes over
    delta_deplete = 1e-4   # normalised retentate flow scale for the smooth depletion factor

    def local_permeate_composition(x, P_ret, P_perm, n_iter=30):
        # Composition of the gas crossing the membrane at each point: y_i = flux_i / sum(flux),
        # with flux_i = Q_i * (P_ret * x_i - P_perm * y_i). Solved by damped fixed-point iteration.
        # This is the correct permeate composition where the permeate flow is zero (closed end, no sweep).
        y_loc = x.copy()
        for _ in range(n_iter):
            flux = permeance_arr[:, None] * np.maximum(P_ret * x - P_perm * y_loc, 0.0)
            y_new = flux / np.maximum(flux.sum(axis=0), 1e-30)
            y_loc = 0.5 * y_loc + 0.5 * y_new
        return y_loc

    def permeate_composition(flow_p, x, P_ret, P_perm, eps=eps_flow):
        # Smooth permeate composition: equals flow_p / sum(flow_p) when the permeate flow is large,
        # and tends to the local flux composition when the permeate flow goes to zero.
        # Replaces the hard 1e-6 threshold that made the ODE right-hand side discontinuous near z = L.
        # flow_p: positive normalised component permeate flows, shape (J, n_pts)
        y_loc = local_permeate_composition(x, P_ret, P_perm)
        return (flow_p + eps * y_loc) / (flow_p.sum(axis=0) + eps)

    '''---------------------------------------------------------------###
    ###---------- Non discretised solution for initial guess ----------###
    ###---------------------------------------------------------------'''

    def approx_mass_balance(vars, area):
        # Naming: N = feed end (z=0), 0 = sweep end (z=L)
        # Counter-current: feed inlet (x_N) faces permeate outlet (y_N), retentate outlet (x_0) faces sweep inlet (y_0)

        J = len(Membrane["Feed_Composition"])

        x_N = Membrane["Feed_Composition"]
        y_0 = Membrane["Sweep_Composition"]
        cut_r_N = Membrane["Feed_Flow"]/Membrane["Total_Flow"]
        cut_p_0 = Membrane["Sweep_Flow"]/Membrane["Total_Flow"]

        Qp_0 = Membrane["Sweep_Flow"]

        x_0 = vars [0:J]
        y_N = vars [J:2*J]
        cut_r_0 = vars[-2]
        cut_p_N= vars[-1]

        Qp_N = Membrane["Total_Flow"] * cut_p_N

        eqs = [0]*(2*J+2)

        eqs[0] = sum(x_0) - 1
        eqs[1] = sum(y_N) - 1

        for i in range(J):
            eqs[i+2] = ( x_N[i] * cut_r_N - x_0[i] * cut_r_0 + y_0[i] * cut_p_0 - y_N[i] * cut_p_N )

        for i in range (J):
            # log-mean partial pressure difference for component i, used in the mass transfer equation
            pp_diff_feed_end  = Membrane["Pressure_Feed"] * x_N[i] - Membrane["Pressure_Permeate"] * y_N[i]
            pp_diff_sweep_end = Membrane["Pressure_Feed"] * x_0[i] - Membrane["Pressure_Permeate"] * y_0[i]

            if (pp_diff_feed_end / (pp_diff_sweep_end + epsilon) + epsilon) >= 0:
                ln_term = math.log((pp_diff_feed_end) / (pp_diff_sweep_end + epsilon) + epsilon)
            else:
                ln_term = epsilon

            dP = (pp_diff_feed_end - pp_diff_sweep_end) / ln_term  # log-mean partial pressure difference

            eqs[i+2+J] = 1 - ( area * dP * Membrane["Permeance"][i] ) / ( y_N[i] * Qp_N - y_0[i] * Qp_0 +epsilon)

        return eqs

    def approx_shooting_guess(area):

        J = len(Membrane["Feed_Composition"])
        approx_guess = [1/J]*J * 2 + [0.5] * 2

        approx_sol = least_squares(
            approx_mass_balance,
            approx_guess,
            args=(area,),
            bounds=(0,1),
            xtol=1e-6,
            ftol=1e-6
            )

        return approx_sol


    '''----------------------------------------------------------###
    ###--------------------- BVP Solver ------------------------###
    ###----------------------------------------------------------'''

    def membrane_odes(z, var):
        u_x = np.maximum(var[:J], 1e-10)
        flow_p = np.maximum(-var[J:2*J], 0.0)   # positive permeate component flows
        P_ret = np.maximum(var[2*J],   1e3)   # minimum 0.01 bar
        P_perm  = np.maximum(var[2*J+1], 1e3)   # minimum 0.01 bar

        A      = Fibre_Dimensions["D_out"] * math.pi * Fibre_Dimensions["Number_Fibre"]
        Ttot   = Membrane["Total_Flow"]

        sum_ux = np.sum(u_x, axis=0)
        x = u_x / sum_ux                                       # retentate mole fractions (u_x > 0 by the clip above)
        y = permeate_composition(flow_p, x, P_ret, P_perm)     # smooth permeate mole fractions, no threshold

        driving_force = P_ret * x - P_perm * y

        # smooth suppression of permeation as the retentate empties
        deplete_factor = sum_ux**2 / (sum_ux**2 + delta_deplete**2)   # about 1 for normal flows, goes to 0 as sum_ux goes to 0
        driving_force = driving_force * deplete_factor

        du_x_dz = -(permeance_arr[:, None] * A / Ttot) * driving_force # change in retentate flow profile
        du_y_dz = -du_x_dz # change in permeate flow profile (equal and opposite to retentate)

        # retentate pressure drops in +z direction
        dP_feed_dz = -pressure_drop_retentate(x, sum_ux * Ttot, P_ret)
        # permeate pressure rises in +z direction (flows in -z)
        dP_perm_dz = +pressure_drop_permeate(y, flow_p.sum(axis=0) * Ttot, P_perm)


        return np.concatenate([du_x_dz, du_y_dz,
                               dP_feed_dz[None, :],
                               dP_perm_dz[None, :]], axis=0)

    def bc(ya, yb):
        feed_norm        =  Membrane["Feed_Composition"]  * Membrane["Feed_Flow"]  / Membrane["Total_Flow"]
        sweep_norm       = -Membrane["Sweep_Composition"] * Membrane["Sweep_Flow"] / Membrane["Total_Flow"]
        feed_pressure    =  Membrane["Pressure_Feed"]
        perm_pressure    =  Membrane["Pressure_Permeate"]
        return np.concatenate([
            ya[:J]    - feed_norm,
            yb[J:2*J] - sweep_norm,
            [ya[2*J]   - feed_pressure],    # retentate pressure at z=0, index 2*J
            [ya[2*J+1] - perm_pressure],    # permeate  pressure at z=0 (outlet), index 2*J+1
        ])

    def solve_at_length(L_end):
        # Solve the BVP for a module of length L_end with the same fibres (same fibre count, same area per unit length).
        # Boundary conditions sit at z = 0 and z = L_end, so a shorter module has its retentate outlet
        # and permeate closed end (or sweep inlet) at L_end.
        area_end = Membrane["Area"] * L_end / L

        sol = approx_shooting_guess(area_end) #conducts a simplified mass balance to get an initial guess
        x_ret_out_approx  = sol.x[:J]      # retentate outlet composition (z=L_end)
        y_perm_out_approx = sol.x[J:2*J]   # permeate outlet composition (z=0)
        cut_r_out         = sol.x[-2]
        cut_p_out         = sol.x[-1]
        U_x_zL_approx =  x_ret_out_approx  * cut_r_out   # normalised retentate flows at z=L_end
        U_y_z0_approx = -y_perm_out_approx * cut_p_out   # normalised permeate flows at z=0

        U_x_feed_norm  =  Membrane["Feed_Composition"]  * Membrane["Feed_Flow"]  / Membrane["Total_Flow"]
        U_y_sweep_norm = -Membrane["Sweep_Composition"] * Membrane["Sweep_Flow"] / Membrane["Total_Flow"]

        # Initialise the solution on a coarse grid, using the approximated values from the simplified mass balance
        x_init = np.linspace(0, L_end, 10)
        n_init = x_init.shape[0]

        U_x_init = np.zeros((J, n_init))
        U_y_init = np.zeros((J, n_init))
        for i in range(J):
            U_x_init[i, :] = np.linspace(U_x_feed_norm[i], U_x_zL_approx[i], n_init) # linear profile between feed and retentate outlet
            U_y_init[i, :] = np.linspace(U_y_z0_approx[i], U_y_sweep_norm[i], n_init) # linear profile between permeate outlet and sweep inlet

        P_perm_init = np.full(n_init, Membrane["Pressure_Permeate"]) #initial guess of constant permeate pressure, will be updated by solver to account for pressure drop
        P_feed_init = np.full(n_init, Membrane["Pressure_Feed"])

        y_init = np.vstack([U_x_init, U_y_init, P_feed_init, P_perm_init])

        ### BVP SOLVER ###
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                  module='scipy.integrate._bvp')
            return solve_bvp(membrane_odes, bc, x_init, y_init, tol=1e-4, max_nodes=1000)

    '''----------------------------------------------------------###
    ###------------- Solve, with retentate depletion -------------###
    ###----------------------------------------------------------'''

    depletion_threshold = 1e-4 * Membrane["Total_Flow"]   # mol/s, retentate flow below which the retentate counts as depleted

    def retentate_flow(sol):
        return np.sum(sol.y[:J, :], axis=0) * Membrane["Total_Flow"]   # mol/s along the mesh

    def is_valid(sol):
        # converged and retentate outlet flow still at or above the depletion threshold
        return sol.success and retentate_flow(sol)[-1] >= depletion_threshold

    bvp_sol = solve_at_length(L)
    L_eff = L
    depleted = np.any(retentate_flow(bvp_sol) < depletion_threshold)

    if depleted:
        # Retentate empties inside the module. Find the event length by re-solving with the
        # boundary conditions at the trial end point (Option A: same fibres, shorter length).
        Qr_first = retentate_flow(bvp_sol)
        i_cross  = int(np.argmax(Qr_first < depletion_threshold))
        L_guess  = bvp_sol.x[max(i_cross, 1)]           # first estimate from the full-length solution

        L_lo, sol_lo = 0.0, None                         # longest length known to be valid
        L_hi = L                                         # shortest length known to be invalid
        trial = L_guess
        while sol_lo is None and trial > 1e-6 * L:       # find a valid lower bound, stepping down from the estimate
            s = solve_at_length(trial)
            if is_valid(s):
                L_lo, sol_lo = trial, s
            else:
                L_hi = min(L_hi, trial)
                trial *= 0.8

        if sol_lo is None:
            print("WARNING: retentate depletion detected but no valid shorter module was found. Returning the full-length solution.")
            Membrane["Effective_Area"]           = Membrane["Area"]
            Fibre_Dimensions["Effective_Length"] = L
        else:
            for _ in range(40):                          # bisection between valid L_lo and invalid L_hi
                if (L_hi - L_lo) / L < 1e-4:
                    break
                L_mid = 0.5 * (L_lo + L_hi)
                s = solve_at_length(L_mid)
                if is_valid(s):
                    L_lo, sol_lo = L_mid, s
                else:
                    L_hi = L_mid
            bvp_sol, L_eff = sol_lo, L_lo
            Membrane["Effective_Area"]           = Membrane["Area"] * L_eff / L
            Fibre_Dimensions["Effective_Length"] = L_eff
            print(f"Retentate depleted: module cut at norm_z={L_eff/L:.4f}. "
                  f"Effective area {Membrane['Effective_Area']:.2f} m2 (input area {Membrane['Area']:.2f} m2). "
                  f"Results below are for the effective area.")
    else:
        Membrane["Effective_Area"]           = Membrane["Area"]
        Fibre_Dimensions["Effective_Length"] = L

    # Report solver failure
    if not bvp_sol.success:
        print(f"WARNING: solve_bvp did not converge (status {bvp_sol.status}): {bvp_sol.message}")
        print("         Results below may be wrong. Check the profile or adjust Area, flows or pressures.")

    y_sol = bvp_sol.y                          # shape (2J+2, n_points) on adaptive mesh
    z_adaptive = bvp_sol.x                     # adaptive mesh chosen by solver

    U_x_profile = y_sol[:J, :]  * Membrane["Total_Flow"]
    U_y_profile = y_sol[J:2*J, :] * Membrane["Total_Flow"]

    Qr_profile =  np.sum(U_x_profile, axis=0)
    Qp_profile = -np.sum(U_y_profile, axis=0)

    P_feed_profile = y_sol[2*J,   :]   # retentate pressure
    P_perm_profile = y_sol[2*J+1, :]   # permeate  pressure

    # Retentate mole fractions (retentate flow stays positive, guarded against division by zero)
    x_profiles = np.maximum(U_x_profile, 0.0) / np.maximum(Qr_profile, 1e-10)

    # Permeate mole fractions from the same smooth expression as the ODEs,
    # so the profile shows the local flux composition at the closed end instead of 0
    y_profiles = permeate_composition(np.maximum(-y_sol[J:2*J, :], 0.0), x_profiles,
                                      np.maximum(P_feed_profile, 1e3), np.maximum(P_perm_profile, 1e3),
                                      eps=1e-15)   # tiny eps: exact flow ratio wherever flow > 0, local flux composition only where flow = 0

    z_norm = z_adaptive / L   # normalised by the input length, so a cut module ends at norm_z < 1
    data = {
        "norm_z":   z_norm,
        **{f"x{i+1}": x_profiles[i, :] for i in range(J)},
        **{f"y{i+1}": y_profiles[i, :] for i in range(J)},
        "Qr":       Qr_profile,
        "Qp":       Qp_profile,
        "P_feed":   P_feed_profile,
        "P_perm":   P_perm_profile,
    }

    profile = pd.DataFrame(data)

    x_ret  = profile.iloc[-1][[f"x{i+1}" for i in range(J)]].values
    y_perm = profile.iloc[0][[f"y{i+1}" for i in range(J)]].values
    Qr     = profile.iloc[-1]["Qr"]
    Qp     = profile.iloc[0]["Qp"]

    # One summary of hydraulics after the solve
    v_shell_in = (Membrane["Feed_Flow"] * R_GAS * Membrane["Temperature"] / Membrane["Pressure_Feed"]) / Fibre_Dimensions["Number_Module"] / A_shell
    print("Pressure drop across the module:")
    #in retentate, compare pressure at z=0 and z=L
    print(f"Retentate pressure drop: {abs(profile['P_feed'].iloc[0] - profile['P_feed'].iloc[-1]):.2f} Pa (shell inlet velocity {v_shell_in:.3f} m/s)")
    #in permeate, compare pressure at z=0 and z=L
    print(f"Permeate pressure drop: {abs(profile['P_perm'].iloc[-1] - profile['P_perm'].iloc[0]):.2f} Pa")

    # Pressure plot 
    if Membrane.get("Plot_Profiles", False):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(profile["norm_z"], profile["P_feed"], label="Retentate Pressure (Pa)", color='red')
        ax1.set_xlabel("Normalized Module Length")
        ax1.set_ylabel("Retentate Pressure (Pa)", color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax2 = ax1.twinx()
        ax2.plot(profile["norm_z"], profile["P_perm"], label="Permeate Pressure (Pa)", color='blue')
        ax2.set_ylabel("Permeate Pressure (Pa)", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        plt.title("Pressure Profiles Along the Module")
        plt.show()

    CC_ODE_results = x_ret, y_perm, Qr, Qp
    return CC_ODE_results, profile