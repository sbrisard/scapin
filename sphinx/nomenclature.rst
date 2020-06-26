############
Nomenclature
############

- :math:`d`: dimension of the physical space (typically :math:`d=2, 3`)
- :math:`\Omega`: :math:`d`-dimensional unit-cell
- :math:`L_1,\ldots, L_d`: dimensions of the unit-cell:
  :math:`\Omega=(0, L_1)\times(0, L_2)\times\cdots\times(0, L_d)`
- :math:`\lvert\Omega\rvert=L_1L_2\cdots L_d`: volume of the unit-cell
- :math:`\tuple{n}`: :math:`d`-dimensional tuple of integers
  :math:`\tuple{n}=(n_1, n_2, \ldots, n_d)`
- :math:`\tuple{N}=(N_1, N_2, \ldots, N_d)`: size of the simulation grid
- :math:`\lvert N\rvert=N_1N_2\cdots N_d`: total number of cells
- :math:`h_i=L_i/N_i`: size of the cells (:math:`i=1, \ldots, d`)
- :math:`\cellindices=\{0, \ldots, N_1-1\}\times\{0, \ldots, N_2-1\}\times\{0,
  \ldots, N_3-1\}`: set of cell indices
- :math:`\Omega_{\tuple{p}}`: cells of the simulation grid
  (:math:`\tuple{p}\in\cellindices`)
