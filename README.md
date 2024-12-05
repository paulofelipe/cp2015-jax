# Caliendo and Parro (2015) - JAX

This repository presents the implementation of the solution of the model presented in Caliendo and Parro (2015) using the JAX library of Python.

The code replicates Table 2 of estimated welfare effects for NAFTA.


Run the cp2015.py script to get the results.

# Data format

The data is a dictionary with the following keys:

* regions - A list of size N with the names of the regions in the same order as they appear in the data.
* sectors - A list of size J with the names of the sectors in the same order as they appear in the data.
* X_nj - An array of size N x J with the values of the total expenditure in each region and sector (gross of import taxes).
* D_n - An array of size N with the values of the defict in each region.
* wL_n - An array of size N with the values of the value added in each region.
* pi_nij - An array of size N x N x J with the shares in trade by importer, exporter and sector. 
* tau_nij - An array of size N x N x J with the initial tariffs by importer, exporter and sector.
* tau_nij_bln - An array of size N x N x J with the tariffs by importer, exporter and sector for the baseline scenario.
* tau_nij_cfl - An array of size N x N x J with the tariffs by importer, exporter and sector for the counterfactual scenario.
* theta_j - An array of size J with the values of the trade elasticity for each sector.
* gamma_nkj - An array of size N x J x J with the shares of input usage in the total output by country, input sector and output sector.
* gamma_nj - An array of size N x J with the share of value added in the total output by country and sector.
* alpha_nj - An array of size N x J with the share of each sector in final demand by country and sector.
