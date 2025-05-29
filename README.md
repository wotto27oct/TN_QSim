# TN_QSim

`repetition_code_simulation.py` is a program for simulating repetition code.

`repetition_code_simulation_with_GTA` is a program for simulating repetition code with generalized twirling approximation.

`thin_surface_code_simulation.py` is a program for simulating 3xn repetition code.

## required libraries
`numpy`, `jax`, `jaxlib`, `opt_einsum`, `kahypar`, `tqdm`, `optuna`, `autoray`, `tensornetwork`, `cotengra` is needed.

## parameters

- `dqnum` is a number of data qubits for repetition code.
- `ncolumn` is a number of column for 3xn surface code.
- `nround` is a number of syndrome measurement rounds.
- `theta` is a strength of rotation noise.
- `gamma` is a coupling strenght to the bath.
- `tau` is a period of gate time.
- `temp` is a temperature.
- `spread` is a rotation angle for leakage spreading.
- `rtype` is a type for leakage removal strategies.
- `init_state` is a initial state, 0 or 1.
- `seed` is a seed for randomization.
- `nshot` is a number of shots to average.
- `threshold_err` is a threshold for truncating small singular values.


## usage of repetition_code_simulation.py
```
python3 repetition_code_simulation.py ${dqnum} ${nround} ${theta} ${gamma} ${tau} ${temp} ${spread} ${rtype} ${init_state} ${seed} ${nshot} ${threshold_err} ${save_folder}
```

## usage of repetition_code_simulation_with_GTA.py
```
python3 repetition_code_simulation_with_GTA.py ${dqnum} ${nround} ${theta} ${gamma} ${tau} ${temp} ${spread} ${rtype} ${init_state} ${seed} ${nshot} ${threshold_err} ${save_folder}
```

## usage of thin_surface_code_simulation.py
```
python3 thin_surface_code_simulation_with.py ${ncolumn} ${nround} ${theta} ${gamma} ${tau} ${temp} ${spread} ${rtype} ${init_state} ${seed} ${nshot} ${threshold_err} ${save_folder}
```

### usage example
```
python3 repetition_code_simulation_with_GTA.py 11 11 0.1 0.1 1.0 100 0.0 noreset 0 0 10 8 None
```