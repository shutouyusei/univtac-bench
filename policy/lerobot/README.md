# lerobot bridge

Stock SmolVLA and the `tacforcing` plugin (from `so101-tactile/plugins`) trained in
the `lerobot` conda env and evaluated in UniVTAC through an HTTP server, so the
sim and the real robot run one policy code base.

```
process_data.py      CLI: HDF5 -> LeRobot v3 dataset            (lerobot env)
train.sh             lerobot-train recipes: smolvla | tacforcing (lerobot env)
server.py            CLI: inference server                       (lerobot env)
deploy_policy.py     UniVTAC BasePolicy adapter                  (Isaac env, no lerobot import)
deploy_smolvla.yml, deploy_tacforcing.yml

tactile/ftp1_encoder.py   frozen FTP-1 GelSight Mini encoder (timm + safetensors, no openpi)
convert/schema.py         dataset keys and dims, feature schema, instruction, cameras
convert/hdf5.py           reading one HDF5 episode (joints, camera and fingertip frames)
convert/transforms.py     state/action pairing, resize, colour order, fingertip stacking
convert/pipeline.py       encode_episode -> write_episode -> source_metadata.json
bridge/wire.py            array <-> JSON encoding shared by both sides
bridge/observation.py     Isaac side: trim a UniVTAC observation to the bridge inputs
bridge/client.py          Isaac side: start/connect, retry, validate, fail fast
bridge/batch.py           lerobot side: wire observation -> policy frame
bridge/local_policy.py    lerobot side: checkpoint + processors + tactile encoder
bridge/app.py             lerobot side: FastAPI routes
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

Camera and fingertip frames keep the channel order UniVTAC records them in
(the simulator's RGB; the HDF5 stream decodes back to it), so training data,
live inference and FTP-1's own UniVTAC pipeline all see the same colours.
Fingertip images reach the encoder at 224x224 in [-1, 1]. Raw fingertip images are stored only with
`--tactile-images` (lerobot would make them policy cameras).

## One loop on insert_hole

```bash
LR=~/miniforge3/envs/lerobot/bin/python
$LR policy/lerobot/process_data.py insert_hole clean51 50 --fps 60 --embedding cls
bash policy/lerobot/train.sh smolvla    policy/lerobot/data/local/insert_hole-clean51-50 300
bash policy/lerobot/train.sh tacforcing policy/lerobot/data/local/insert_hole-clean51-50 300
# UniVTAC env, from the repo root (OMNI_KIT_ACCEPT_EULA=YES when stdin is not a terminal):
bash eval_policy.sh insert_hole clean51 lerobot/deploy_smolvla    0 --total_num 2 --max_seed 1000003 --headless
bash eval_policy.sh insert_hole clean51 lerobot/deploy_tacforcing 0 --total_num 2 --max_seed 1000003 --headless
```

`deploy_*.yml` names the `pretrained_model` directory; the server reads the
policy type and the saved preprocessor (normalisation, camera renaming) from it.
Set `lerobot_port` to reuse a server started by hand:
`$LR policy/lerobot/server.py --port 10800`.

## When the server fails

`scripts/eval_policy.py` catches every exception from `policy.eval`, writes it
with a traceback to `eval_result/.../log.log`, marks the seed `error` and moves
on. `bridge/client.py` makes such failures immediate and specific:

- connection refused/reset/timeout: retried `lerobot_retries` times (default 3)
  with a short backoff, each attempt logged as `[lerobot-bridge] ...` on stderr;
- server process gone: raised at once with its exit code, also during startup;
- server-side exception: raised with the server's own traceback, no retry;
- malformed answer (missing action, wrong shape, NaN): raised before Isaac sees it;
- after any of these the client is `dead` and every later call raises at once,
  so the remaining seeds fail in milliseconds instead of one timeout each.

Fix the cause and rerun the eval; `--max_seed` bounds the run either way.

## Notes

- Each eval step is one `policy.select_action`: SmolVLA runs its 50-action chunk
  open loop, tacforcing streams 5-action blocks and re-reads tactile per block.
- `fps 60` matches `save: 60` in `task_config/clean51.yml`.
- Short runs here verify the pipeline; success rates from them are not results.
