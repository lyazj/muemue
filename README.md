# Quantum state tomography with muons

https://arxiv.org/abs/2411.12518

## Environment

`MadGraph5_aMC@NLO 3.5` (https://launchpad.net/mg5amcnlo/3.0/3.5.x)

CERN `ROOT 6.28+` (https://root.cern.ch/)

Python3 with: `uproot5` (with compatible `awkward` and `numpy`), `matplotlib`

## Structure

`qe`: QE analysis programs

### Entries for direct invocation

`make_muemue_examples_*GeV.py`: Script to run MG5

`plot_muemue_kinematics.py`: Plot kinematic distributions and generate TXT data files for QE studies

### Entries for underlying implementation

`cards`: MG5 cards for MC event generation

`MG5*.py`: Implementation of MG5 interface

## Instruction

* Run `make_muemue_examples_*GeV.py` to generate MC events (the number of events and other parameters can be adjusted by modifying this script)

* Run `plot_muemue_kinematics.py` to make kinematic plots and acquire TXT data files for subsequent QE analysis
