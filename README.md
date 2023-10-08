# ROMA by kimdohyun

내가 원본 코드에서 바뀐 부분을 여기에 설명한다. (추후, 코드를 아예 바꿔버리자.)

- `src/envs/gfootball/__init__.py`에서 gfootball 관련 모두 주석처리 (어차피 안씀)
- `.git` 폴더를 `..git` 또는 다른 것으로 변경 (아니면 sacred에서 에러 남)
- `/usr/local/lib/python3.8/dist-packages/tensorboardX/writer.py`에서 `warning: Embedding dir exists, did you set global_step for add_embedding()?` warning을 주석처리함.
- memory 작은 이슈로, `src/config/algs/qmix_smac_latent.yaml`의 runner를 "episode"로 (기존 "parallel"), batch_size_run을 1로(기존 8), `src/config/default.yaml`의 batch_size 1로 (기존 32) 설정함
- `src/utils/logging.py` 에서 56번 Line np.mean에서 th.mean으로 변경
<details>
    <summary>
    Traceback 펼치기    
    </summary>
```
  File "/pymarl/src/utils/logging.py", line 55, in print_recent_stats
    item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
  File "<__array_function__ internals>", line 5, in mean
  File "/usr/local/lib/python3.8/dist-packages/numpy/core/fromnumeric.py", line 3419, in mean
    return _methods._mean(a, axis=axis, dtype=dtype,
  File "/usr/local/lib/python3.8/dist-packages/numpy/core/_methods.py", line 162, in _mean
    arr = asanyarray(a)
  File "/usr/local/lib/python3.8/dist-packages/numpy/core/_asarray.py", line 171, in asanyarray
    return array(a, dtype, copy=False, order=order, subok=True)
  File "/usr/local/lib/python3.8/dist-packages/torch/_tensor.py", line 1030, in __array__
    return self.numpy()
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

</details>

```shell
# run docker with STDIN and pesudo tty
sudo docker run \
    --name pymarl_primi \
    -v `pwd`:/pymarl \
    --gpus all \
    -it pymarl:1.0 /bin/bash

python3 src/main.py \
--config=qmix_smac_latent \
--env-config=sc2 \
with \
agent=latent_ce_dis_rnn \
env_args.map_name=MMM2 \
t_max=20050000
```

------------

In ROMA's ICML paper, we use [an old version of the SMAC benchmark](https://arxiv.org/pdf/1902.04043v1.pdf) for both ROMA and the baselines (QMIX, COMA, IQL, MAVEN, QTRAN), and their performance are different from that can be achieved by the latest version.

# ROMA: Multi-Agent Reinforcement Learning with Emergent Roles

## Note
 This codebase accompanies the paper submission "ROMA: Multi-Agent Reinforcement Learning with Emergent Roles" ([ROMA website](https://sites.google.com/view/romarl)), and is based on  [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) codebases which are open-sourced.

The implementation of the following methods can also be found in this codebase, which are finished by the authors of [PyMARL](https://github.com/oxwhirl/pymarl):

- [**ROMA**: ROMA: Multi-Agent Reinforcement Learning with Emergent Roles](https://arxiv.org/abs/2003.08039)
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

If you want to run the environments we designed, move all the SC2 maps in `src/envs/starcraft2/map/designed/` to `3rdparty/StarCraftII/Maps/SMAC_Maps/`.
It is worth noting that `bane_vs_bane1` corresponds to `6z4b`, `zb_vs_sz` corresponds to `10z5b_vs_2s3z`, and `sz_vs_zb` 
corresponds to `6s4z_vs_10b30z` in the paper. 

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

```shell
python3 src/main.py \
--config=qmix_smac_latent \
--env-config=sc2 \
with \
agent=latent_ce_dis_rnn \
env_args.map_name=MMM2 \
t_max=20050000
```

To test other maps, add parameters

```shell
h_loss_weight=5e-2
var_floor=1e-4
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

To run experiments using the Docker container:

```shell
bash run.sh $GPU python3 src/main.py \
--config=qmix_smac_latent \
--env-config=sc2 \
with \
agent=latent_ce_dis_rnn \
env_args.map_name=MMM2 \
t_max=20050000
```


All results will be stored in the `Results` folder.



## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.
