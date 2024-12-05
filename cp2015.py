"""
This script implements the baseline and counterfactual NAFTA scenarios of the Caliendo
and Parro (2015) model. The model is solved using the algorithm described in the
Appendix C of the paper. The dataset used in this script is stored in the cp2015_data.pkl
file. 

In order to increase the efficiency of the code, we use the JAX library and JIT (Just-In-Time)
compilation. JIT compiles the function in an optimized way for execution on CPU or GPU.
"""

import pickle
import jax
import jax.numpy as jnp

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)
# Alias for jnp.newaxis
nax = jnp.newaxis
# Avoid displaying scientific notation
jnp.set_printoptions(suppress=True)

data = pickle.load(open("cp2015_data.pkl", "rb"))


def run_cp2015(data):
    """
    Run the baseline and counterfactual NAFTA scenarios of the Caliendo and Parro (2015)
    model.

    Parameters:
    data: dictionary with the dataset. Inspect cp2015_data.pkl for more information
    about the dataset format.
    """

    # tau_nij1 are the tariffs for the baseline scenario (No NAFTA)
    data["tau_nijp"] = data["tau_nij_bln"]
    bln = solve_model(data)

    # tau_nij1_cfl are the tariffs for the counterfactual scenario (NAFTA)
    data["tau_nijp"] = data["tau_nij_cfl"]
    cfl = solve_model(data)

    return bln, cfl


def solve_model(data):
    """
    Solve the Caliendo and Parro (2015) model using the algorithm described in the
    Appendix C of the paper.

    Parameters:
    data: dictionary with the dataset. Inspect cp2015_data.pkl for more information
    about the dataset format.
    """

    # Regions and sectors present in the dataset
    regions = data["regions"]
    sectors = data["sectors"]

    N = len(regions)  # Number of regions
    J = len(sectors)  # Number of sectors

    # Get some variables from the dataset
    X_njp = jnp.array(data["X_nj"])
    D_n = jnp.array(data["D_n"])
    tau_nijp = jnp.array(data["tau_nijp"])
    d_nij_hat = jnp.ones((N, N, J))

    # Get the parameters from the dataset
    theta_j = jnp.array(data["theta_j"])
    pi_nij = jnp.array(data["pi_nij"])
    gamma_nkj = jnp.array(data["gamma_nkj"])
    gamma_nj = jnp.array(data["gamma_nj"])
    alpha_nj = jnp.array(data["alpha_nj"])
    wL_n = jnp.array(data["wL_n"])
    tau_nij = jnp.array(data["tau_nij"])

    # Model solution

    # Step 1: Guess a vector of wages w_n_hat (w_1_hat, ..., w_N_hat), matrix of
    # total expenditures X_njp, and matrix of prices P_nj_hat
    w_n_hat0 = jnp.ones(N)

    # A guess for P_nj_hat will be used in step 2
    P_nj_hat0 = jnp.ones((N, J))

    # Compute change in trade costs (kappa_hat_nij)
    kappa_hat_nij = (1 + tau_nijp) / (1 + tau_nij) * d_nij_hat

    wcriteria = 1e5
    iter = 0
    while wcriteria > 1e-7:

        # Step 2: Use equilibrium conditions (10) and (11) to solve for prices in each
        # sector and each country, P_nj_hat (P_1j_hat, ..., P_Nj_hat) and input costs
        # c_nj_hat (c_1j_hat, ..., c_Nj_hat) consistent with the vector of wages w_n_hat

        # Iterate until convergence
        pcriteria = 1e5
        while pcriteria > 1e-6:
            c_nj_hat, P_nj_hat = solve_P_c_jit(
                w_n_hat0,
                P_nj_hat0,
                pi_nij,
                kappa_hat_nij,
                gamma_nj,
                gamma_nkj,
                theta_j,
            )
            pcriteria = jnp.linalg.norm(P_nj_hat - P_nj_hat0)
            P_nj_hat0 = P_nj_hat.copy()

        # Step 3: Step 3: Use the information on pi_nij and theta_j together with the
        # solutions to P_nj_hat(w) and c_nj_hat(w) from step 2 and solve for pi_nijp using
        # (12)
        pi_nijp = pi_nij * (
            kappa_hat_nij * c_nj_hat[nax, :, :] / P_nj_hat[:, nax, :]
        ) ** (-theta_j[nax, nax, :])

        # Step 4: Given pi_nijp(w) from step 3, the new tariff vector tau_nijp, and data
        # for gamma_nj, gamma_nkj, alpha_nj, solve for total expenditure in each sector
        # j and country n, X_njp(w), consistent with the vector of wages w_n_hat in the
        # following way. X(w) = Ω^-1(w) Δ(W), where Ω(w) and Δ(W) are defined in the
        # paper.
        # Δ(W)
        X_njp = solve_X_jit(
            w_n_hat0,
            gamma_nkj,
            pi_nijp,
            tau_nijp,
            D_n,
            alpha_nj,
            wL_n,
        )

        # Step 5: Substitute pi_nijp(w), X_njp(w), tau_nijp, and D_nt and obtain:
        # (Check the trade balance per country. Equation C2 in the paper)
        res_tb = (
            (pi_nijp / (1 + tau_nijp) * X_njp[:, nax, :]).sum(axis=(1, 2))
            - D_n
            - (pi_nijp / (1 + tau_nijp) * X_njp[:, nax, :]).sum(axis=(0, 2))
        )

        # Step 6: Verify if equation (C2) holds. If not, we adjust our guess of ˆw and proceed to step 1 again until equilibrium
        # condition (C2) is obtained.
        vfactor = -0.5
        w_n_hat = w_n_hat0 * (1 + vfactor * res_tb / wL_n)
        wcriteria = jnp.linalg.norm(w_n_hat - w_n_hat0)
        w_n_hat0 = w_n_hat
        if iter % 10 == 0 or wcriteria < 1e-7:
            print("Iteration: ", iter, " Convergence criteria: ", wcriteria)
        iter += 1

    P_n_hat = (P_nj_hat**alpha_nj).prod(axis=1)
    wp_n_hat = w_n_hat0 / P_n_hat
    I_np = (
        w_n_hat0 * wL_n
        + (tau_nijp * pi_nijp / (1 + tau_nijp) * X_njp[:, nax, :]).sum(axis=(1, 2))
        + D_n
    )

    variables = {
        "c_nj_hat": c_nj_hat,
        "P_nj_hat": P_nj_hat,
        "pi_nijp": pi_nijp,
        "X_njp": X_njp,
        "I_np": I_np,
        "w_n_hat": w_n_hat,
        "P_n_hat": P_n_hat,
        "wp_n_hat": wp_n_hat,
        "tau_nijp": tau_nijp,
        "tau_nij": tau_nij,
    }

    return variables


def solve_P_c(
    w_n_hat0,
    P_nj_hat0,
    pi_nij,
    kappa_hat_nij,
    gamma_nj,
    gamma_nkj,
    theta_j,
):
    """
    Solve for prices P_nj_hat and input costs c_nj_hat consistent with the vector of
    wages w_n_hat

    Parameters:
    w_n_hat0: vector of wages
    P_nj_hat0: Array of composite prices by importing country and sector
    pi_nij: Array of trade shares by importing country, exporting country, and sector
    kappa_hat_nij: Array of change in trade costs by importing country, exporting country,
    and sector
    gamma_nj: Share of value added in total output by country and sector
    gamma_nkj: Share of intermediate inputs in total output by importing country, input,
    and sector
    theta_j: trade elasticity by sector
    """
    c_nj_hat = w_n_hat0[:, nax] ** gamma_nj * (P_nj_hat0[:, :, nax] ** gamma_nkj).prod(
        axis=1
    )
    P_nj_hat = (
        (pi_nij * (kappa_hat_nij * c_nj_hat[nax, :, :]) ** (-theta_j)).sum(axis=1)
    ) ** (-1 / theta_j)

    return c_nj_hat, P_nj_hat


solve_P_c_jit = jax.jit(solve_P_c)


def solve_X(
    w_n_hat0,
    gamma_nkj,
    pi_nijp,
    tau_nijp,
    D_n,
    alpha_nj,
    wL_n,
):
    """
    Solve for total expenditure in each country and sector consistent with the vector of
    wages w_n_hat.

    Parameters:
    w_n_hat0: vector of wages
    gamma_nkj: Share of intermediate inputs in total output by importing country, input,
    and sector
    pi_nijp: Array of trade shares by importing country, exporting country, and sector
    tau_nijp: Array of trade costs by importing country, exporting country, and sector
    D_n: Array of trade deficit by country
    alpha_nj: Share of each sector in final demand for each country
    wL_n: Array of labor income by country
    """
    N = w_n_hat0.shape[0]
    J = gamma_nkj.shape[1]
    delta_w = (alpha_nj * (w_n_hat0 * wL_n + D_n)[:, nax]).reshape((N * J))
    # F(w)
    Fp = jnp.zeros((N, J))
    for j in range(J):
        Fptmp = (pi_nijp[:, :, j] / (1 + tau_nijp[:, :, j])).sum(axis=1)
        Fp = Fp.at[:, j].set(Fptmp)

    F = jnp.zeros((N * J, N * J))
    for n in range(N):
        Ftmp = jnp.kron(alpha_nj[n, :].reshape((J, 1)), (1 - Fp[n, :]).reshape((1, J)))
        F = F.at[(n * J) : ((n + 1) * J), (n * J) : ((n + 1) * J)].set(Ftmp)
    # H(w)
    H = jnp.zeros_like(F)
    pi_tilde = pi_nijp / (1 + tau_nijp)
    tmp = pi_tilde.transpose((1, 0, 2)).reshape((N, N * J))
    tmp2 = jnp.zeros((N * J, N * J))
    for i in range(N):
        tmp2 = tmp2.at[i * J : (i + 1) * J, :].set(jnp.tile(tmp[i, :], (J, 1)))
    H = jnp.tile(gamma_nkj.reshape((N * J, J)), (1, N)) * tmp2
    # Ω^(w)
    I = jnp.eye(N * J)
    omega_w = I - F - H
    X_njp = jnp.linalg.solve(omega_w, delta_w).reshape((N, J))

    return X_njp


solve_X_jit = jax.jit(solve_X)


def compute_welfare(bln, cfl):
    """
    Compute the welfare effects of the counterfactual scenario relative to the baseline

    Parameters:
    bln: dictionary with the baseline scenario results
    cfl: dictionary with the counterfactual scenario results
    """

    wp_n_hat = cfl["wp_n_hat"] / bln["wp_n_hat"]
    c_nj_hat = cfl["c_nj_hat"] / bln["c_nj_hat"]
    trade_bln = bln["pi_nijp"] * bln["X_njp"][:, nax, :] / (1 + bln["tau_nijp"])
    trade_cfl = cfl["pi_nijp"] * cfl["X_njp"][:, nax, :] / (1 + cfl["tau_nijp"])
    # Exports in baseline
    E_nij_bln = jnp.transpose(trade_bln, (1, 0, 2))
    # Imports in baseline
    M_nij_bln = trade_bln
    # Exports in counterfactual
    E_nij_cfl = jnp.transpose(trade_cfl, (1, 0, 2))
    # Imports in counterfactual
    M_nij_cfl = trade_cfl

    tot_nij = (
        1
        / bln["I_np"][:, nax, nax]
        * (
            (E_nij_bln * (c_nj_hat[:, nax, :] - 1))
            - (M_nij_bln * (c_nj_hat[nax, :, :] - 1))
        )
    )

    vot_nij = (
        1
        / bln["I_np"][:, nax, nax]
        * (bln["tau_nijp"] * M_nij_bln * (M_nij_cfl / (M_nij_bln + 1e-12) - c_nj_hat))
    )

    tot_total = tot_nij.sum(axis=(1, 2))
    vot_total = vot_nij.sum(axis=(1, 2))
    welfare = tot_total + vot_total

    # convert welfare to pandas DataFrame
    import pandas as pd

    welfare_df = pd.DataFrame(
        {
            "Region": data["regions"],
            "Welfare": jnp.round(welfare * 100, 2),
            "Terms of trade": jnp.round(tot_total * 100, 2),
            "Volume of Trade": jnp.round(vot_total * 100, 2),
            "Real wage": jnp.round((wp_n_hat - 1) * 100, 2),
        }
    )

    return welfare_df


# Solve the baseline and counterfactual scenarios for NAFTA example
bln, cfl = run_cp2015(data)
welfare_df = compute_welfare(bln, cfl)

# Replicate Table 2
welfare_nafta = welfare_df[welfare_df["Region"].isin(["USA", "Canada", "Mexico"])]
print(welfare_nafta)
