# TN_QSim

## required libraries
`numpy`, `jax`, `jaxlib`, `opt_einsum`, `kahypar`, `tqdm`, `optuna`, `autoray`, `tensornetwork`, `cotengra` is needed.
See also `Dockerfile_cuda11.4_kahypar` (but it doesn't contain the installation of `cotengra)

## usage
If you want to execute 3×3 surface code simulation under amplitude damping with strength 0.1,
```
python3 surface.py 3 3 AD 0.1 0 1 0
```
where 3(height), 3(width), AD(type), 0.1(strength), 0(seed), 1(#trial), 0(no meaning)

When carrying 3×3 and 5×5 amplitude dampling noise simulation, cotengra uses cached contraction path.
In other situations, cotengra tries to find the optimal contraction path for a minutes by each calculation
(and once the contraction path is calculated, cache is preserved in ctg_path_cache and reused).

If you want to look how FET method decreases the bond dimension,
```
python3 surface_opt_trun.py 3 3 AD 0.1 0 1 0
```