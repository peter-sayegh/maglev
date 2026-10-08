# Levitation and stability in Maglev Systems
 
Experimental data and code for the paper:
 
> P. A. Sayegh, H. Déo, Y. Abou Rabii and A. Couairon, "Levitation and stability in Maglev systems," *European Journal of Physics* (2026). [doi:10.1088/1361-6404/aeaab7](https://doi.org/10.1088/1361-6404/aeaab7)
 
## About the project
 
We built a scaled-down magnetic levitation train from repolarized flexible magnetic tape and measured how its levitation gap changes as mass is added. A closed-form force model derived from a dipole-layer approximation reproduces the measurements with a single fitted parameter, confirms the passive vertical stability of the repulsive configuration, and illustrates the lateral instability required by Earnshaw's theorem. The model is then scaled up to a full-size Transrapid TR08 vehicle.
 
The project grew out of a classical electrodynamics course at École Polytechnique and is intended as a hands-on way to teach magnetic field geometry, vector potentials, energy methods, and Earnshaw's theorem.
 
## Contents
 
| File | Description |
|---|---|
| `MAGLEV_Code/Maglev_EXPDATA.csv` | Measured levitation gap (mm) versus added mass (g), 25 points. Uncertainties: ±1 g on mass, ±0.5 mm on gap. |
| `MAGLEV_Code/maglev_figures.ipynb` | Jupyter notebook that fits the model to the data and reproduces Figures 2–6 of the paper. |
 
**Note on the CSV format:** the file uses semicolons as separators and commas as decimal marks. Load it with:
 
```python
import pandas as pd
data = pd.read_csv("MAGLEV_Code/Maglev_EXPDATA.csv", sep=";", decimal=",")
```
 
## Reproducing the figures
 
Requirements: Python 3 with NumPy, SciPy, Matplotlib, and Jupyter.
 
```bash
pip install numpy scipy matplotlib jupyter
jupyter notebook MAGLEV_Code/maglev_figures.ipynb
```
 
Run all cells in order. The notebook fits the magnetic pressure parameter to the experimental data, then generates:
 
- Figure 2: vertical field profiles across a rail
- Figure 3: magnetic field lines in the vertical plane
- Figure 4: levitation gap versus added mass, with the fitted model
- Figure 5: scaling to the Transrapid TR08
- Figure 6: potential energy landscape and equilibrium gap
## Citation
 
If you use this data or code, please cite the paper:
 
```bibtex
@article{sayegh2026maglev,
  author  = {Sayegh, Peter A. and D{\'e}o, Hadrien and Abou Rabii, Yazan and Couairon, Arnaud},
  title   = {Levitation and stability in maglev systems},
  journal = {European Journal of Physics},
  year    = {2026},
  doi     = {10.1088/1361-6404/aeaab7}
}
```
 
## License
 
Released under the MIT License. See `LICENSE` for details.
