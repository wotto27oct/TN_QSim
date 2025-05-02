# TN_QSim

## required libraries
`numpy`, `jax`, `jaxlib`, `opt_einsum`, `kahypar`, `tqdm`, `optuna`, `autoray`, `tensornetwork`, `cotengra` is needed.

## usage of repetition_code_simulation_with_GTA.py
```
python3 repetition_code_simulation_with_GTA.py ${dqnum} ${nround} ${theta} ${gamma} ${tau} ${temp} ${spread} ${rtype} ${init_state} ${seed} ${nshot} ${threshold_err} ${save_folder}
```

### usage example
```
python3 repetition_code_simulation_with_GTA.py 11 11 0.1 0.1 1.0 100 0.0 noreset 0 0 10 8 None
```

## usage of RQCsimulation_tensorcore.py

```
python3 RQCsimulation_tensorcore.py ${width} ${height} ${depth} xyt ${mode} ${cseed} ${aseed} ${minimize} ${pseed} ${nworkers} ${gtype} ${max_repeat} ${calc_amp}
```

### 説明

- `width height depth`：RQCの設定．depthは0-index．
- `mode`：FP16TCEC, TF32TCEC, FP16TC, TF32TC, CUBLAS, CUBLAS64F
- `cseed`：RQCのseed，0〜9
- `aseed`：振幅のseed．
- `minimize`：flopsまたはgputime．gputimeだと歪な行列積を減らすようにコストが作られる
- `pseed`：cotengraのpath optimizerのseed．
- `nworkers`：path optimizerでどの程度並列実行するか．
- `gtype`：cotengraの保存先のための変数．
- `max_repeat`：path optimizerを何回走らせるか．
- `calc_amp`：振幅を実際に計算するかどうか．Trueで実際に計算する．

例
```
python3 RQCsimulation_tensorcore.py 6 6 17 xyt CUBLAS64F 0 0 gputime 0 24 penalty_heavy2 128 True
```

## usage of surface
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