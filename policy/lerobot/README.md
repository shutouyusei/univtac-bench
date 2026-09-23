# lerobot bridge

Stock SmolVLA and the `tacforcing` plugin (from `so101-tactile/plugins`) trained in
the `lerobot` conda env and evaluated in UniVTAC through an HTTP server, so the
sim and the real robot run one policy code base.

```
tactile_encoder.py   frozen FTP-1 GelSight Mini encoder (timm + safetensors, no openpi)
process_data.py      HDF5 -> LeRobot v3 dataset with observation.environment_state
train.sh             lerobot-train recipes: smolvla | tacforcing
server.py            lerobot env: checkpoint + processors + encoder, /act = select_action
deploy_policy.py     UniVTAC BasePolicy client (Isaac env, no lerobot import)
deploy_smolvla.yml, deploy_tacforcing.yml
```

Tests: `~/miniforge3/envs/lerobot/bin/python -m pytest tests/policy_lerobot`.

## Data contract

| key | value |
|---|---|
| `observation.images.head` (+ `wrist` for `camera_type: all` tasks) | RGB 256x256 |
| `observation.state` | `joint[:8]` (7 arm + gripper) |
| `action` | `joint[:8]` of the next frame |
| `observation.environment_state` | FTP-1 embedding of left then right `rgb_marker`, 2 x 768 (`cls`) or 2 x 512 (`proj`) |
| `task` | first "seen" instruction of `instructions/<task>.json` |

Fingertip images go to the encoder in BGR at 224x224 in [-1, 1], as FTP-1's own
UniVTAC pipeline feeds them. Raw fingertip images are stored only with
`--tactile-images` (lerobot would make them policy cameras).

## One loop on insert_hole

```bash
LR=~/miniforge3/envs/lerobot/bin/python
$LR policy/lerobot/process_data.py insert_hole clean51 50 --fps 60 --embedding cls
bash policy/lerobot/train.sh smolvla    policy/lerobot/data/local/insert_hole-clean51-50 300
bash policy/lerobot/train.sh tacforcing policy/lerobot/data/local/insert_hole-clean51-50 300
# UniVTAC env, from the repo root:
bash eval_policy.sh insert_hole clean51 lerobot/deploy_smolvla    0 --total_num 2 --max_seed 1000003 --headless
bash eval_policy.sh insert_hole clean51 lerobot/deploy_tacforcing 0 --total_num 2 --max_seed 1000003 --headless
```

`deploy_*.yml` names the `pretrained_model` directory; the server reads the
policy type and the saved preprocessor (normalisation, camera renaming) from it.
Set `lerobot_port` to reuse a server started by hand:
`$LR policy/lerobot/server.py --port 10800`.

## Notes

- Each eval step is one `policy.select_action`: SmolVLA runs its 50-action chunk
  open loop, tacforcing streams 5-action blocks and re-reads tactile per block.
- `fps 60` is provisional; it only fixes the timestamps of the block boundaries
  (`k * block_size / fps`) that tacforcing reads from the dataset.
- Short runs here verify the pipeline; success rates from them are not results.
