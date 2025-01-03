"""
Convert the original data from Caliendo and Parro (2015) to the format used by
this project's code.
"""

import numpy as np
import pickle

N = 31  # Number of regions
J = 40  # Number of sectors


################################################################################
# This part of the code follows the processing in Matlab of the file
# Equilibrium/script.m that is in the code made available by Caliendo and
# Parro (2015).
################################################################################

# Trade flows in 1993
xbilat1993 = np.loadtxt("original_data/xbilat1993.txt", delimiter="\t")
xbilat1993 = xbilat1993 * 1000
xbilat1993_new = np.concatenate((xbilat1993, np.zeros((20 * N, N))))

# Tariffs
# Tariffs in 1993
tau1993 = np.loadtxt("original_data/tariffs1993.txt", delimiter="\t")
# Tariffs in 2005
tau2005 = np.loadtxt("original_data/tariffs2005.txt", delimiter="\t")

tau = np.concatenate((1 + tau1993 / 100, np.ones((20 * N, N))))
taup = np.concatenate((1 + tau2005 / 100, np.ones((20 * N, N))))

# Parameters
G = np.loadtxt("original_data/IO.txt", delimiter="\t")
B = np.loadtxt("original_data/B.txt", delimiter="\t")
GO = np.loadtxt("original_data/GO.txt", delimiter="\t")
T = np.loadtxt("original_data/T.txt", delimiter="\t")
T = np.concatenate((1 / T, np.ones((20,)) * 1 / 8.22))

# Expenditures gross of tariffs
xbilat = xbilat1993_new * tau

# Domestic sales
x = np.zeros((J, N))  # exports
xbilat_domestic = xbilat / tau
for i in range(J):
    x[i, :] = xbilat_domestic[(0 + i * N) : ((i + 1) * N), :].sum(axis=0)
GO = np.maximum(GO, x)
domsales = GO - x

for i in range(J):
    np.fill_diagonal(xbilat[(0 + i * N) : ((i + 1) * N), :], domsales[i, :])

# Calculating X0 Expenditure
Xjn = np.zeros((J, N))
for i in range(J):
    Xjn[i, :] = xbilat[(0 + i * N) : ((i + 1) * N), :].sum(axis=1)
Din = xbilat / Xjn.reshape((N * J, 1))

# Calculating superavits
M = np.zeros((J, N))
E = np.zeros((J, N))
for i in range(J):
    tmp = xbilat[(0 + i * N) : ((i + 1) * N), :] / tau[(0 + i * N) : ((i + 1) * N), :]
    M[i, :] = tmp.sum(axis=1)
    E[i, :] = tmp.sum(axis=0)
Sn = E.sum(axis=0) - M.sum(axis=0)

# Calculating value added
VAjn = GO * B
VAn = VAjn.sum(axis=0)

# Calculating shares in final demand
num = np.zeros((J, N))
for n in range(N):
    num[:, n] = Xjn[:, n] - (
        G[(0 + J * n) : ((n + 1) * J), :]
        * (1 - B[:, n].reshape((1, J)))
        * E[:, n].reshape((1, J))
    ).sum(axis=1)

tariff_revenue = np.zeros(N)
for j in range(J):
    t = tau[(0 + j * N) : ((j + 1) * N), :] - 1
    tariff_revenue += (xbilat[0 + j * N : (j + 1) * N, :] / (1 + t) * t).sum(axis=1)
income = VAn + tariff_revenue - Sn
alphas = num / income
alphas[alphas < 0] = 0
alphas = alphas / alphas.sum(axis=0)

################################################################################
# This part of the code do the transformation of the data to the format used by
# this project's code.
################################################################################
# Regions
regions = [
    "Argentina",
    "Australia",
    "Austria",
    "Brazil",
    "Canada",
    "Chile",
    "China",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "India",
    "Indonesia",
    "Ireland",
    "Italy",
    "Japan",
    "Korea",
    "Mexico",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Portugal",
    "South Africa",
    "Spain",
    "Sweden",
    "Turkey",
    "UK",
    "USA",
    "Row",
]

# Sectors
sectors = [
    "Agriculture",
    "Mining",
    "Food",
    "Textile",
    "Wood",
    "Paper",
    "Petroleum",
    "Chemicals",
    "Plastic",
    "Minerals",
    "Basic metals",
    "Metal products",
    "Machinery n.e.c",
    "Office",
    "Electrical",
    "Communication",
    "Medical",
    "Auto",
    "Other Transport",
    "Other",
    "Electricity",
    "Construction",
    "Retail",
    "Hotels",
    "Land Transport",
    "Water Transport",
    "Air Transport",
    "Aux Transport",
    "Post",
    "Finance",
    "Real State",
    "Renting Mach",
    "Computer",
    "R&D",
    "Other Business",
    "Public",
    "Education",
    "Health",
    "Other services",
    "Private",
]

# Expenditures by region and sector
X_nj = Xjn.T

# Trade deficit by region
# use this if you want run simulations with trade deficits
# different from zero.
# D_n = -Sn
# use this if you want run simulations with trade deficits equal to zero.
D_n = np.zeros(N)

# Total value added by region
wL_n = VAn

# Shares in trade by importer, exporter and sector
pi_nij = Din.reshape((J, N, N)).transpose((1, 2, 0))

# Initial tariffs
tau_nij = (tau - 1).reshape((J, N, N)).transpose((1, 2, 0))

# Tariffs for baseline
tau_nij_bln = tau_nij.copy()

# Tariffs for counterfactual
# The counterfactual is the same as the baseline, except for the tariffs for
# Nafta countries.
tariff_2005 = (taup - 1).reshape((J, N, N)).transpose((1, 2, 0))
tau_nij_cfl = tau_nij_bln.copy()
nafta = ["Canada", "Mexico", "USA"]
nafta_idx = [regions.index(n) for n in nafta]
for n in nafta_idx:
    for i in nafta_idx:
        tau_nij_cfl[n, i, :] = tariff_2005[n, i, :]

# Dispersion of productivity
theta_j = 1 / T

# Share of materials by region, input and output
gamma_nkj = G.reshape((N, J, J)) * (1 - B.T.reshape((N, 1, J)))

# Share of labor by region and sector
gamma_nj = B.T

# Shares in final demand by region and sector
alpha_nj = alphas.T

# Save the data
output_data = {
    "regions": regions,
    "sectors": sectors,
    "X_nj": X_nj,
    "D_n": D_n,
    "wL_n": wL_n,
    "pi_nij": pi_nij,
    "tau_nij": tau_nij,
    "tau_nij_bln": tau_nij_bln,
    "tau_nij_cfl": tau_nij_cfl,
    "theta_j": theta_j,
    "gamma_nkj": gamma_nkj,
    "gamma_nj": gamma_nj,
    "alpha_nj": alpha_nj,
}

pickle.dump(output_data, open("cp2015_data.pkl", "wb"))
