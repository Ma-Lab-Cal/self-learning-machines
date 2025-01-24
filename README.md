## Overview
Simulation of self-learning electronic circuits using open-source SPICE alternatives. 

### Software Requirements
- Conda (tested with 24.1.2)

## Installation Steps
1. **Install Ngspice:** We recommend building from source.
   Ngspice download instructions: https://ngspice.sourceforge.io/download.html

   This project was tested on Ngspice version 43 and the following compilation flags:
   ```--with-ngshared --enable-cider --enable-openmp```
   
  **The installation scripts should include both Sparse and KLU solvers by default. Make neither solver is turned off when installing!**

   For more information, see Chapter 28 of the documentation: https://ngspice.sourceforge.io/docs.html

3. **Set Up the Virtual Environment:** The provided file `self-learning-machines.yml` will create a Conda environment called `circuit-sim` with all libraries EXCEPT `PySpice` installed.
   PySpice must be installed separately using the custom Ma Lab fork. 
   ```bash
   conda env create -f self-learning-machines.yml
   ```

4. **Clone the Ma Lab fork of PySpice:** https://github.com/Ma-Lab-Cal/PySpice  

5. **Manually install the PySpice fork:**  
     ```bash
   conda activate circuit-sim
   pip install -e <PATH-TO-FORKED-PYSPICE>
   ```
