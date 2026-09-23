"""The UniVTAC <-> lerobot env bridge.

Isaac side (no lerobot import): ``observation`` trims a UniVTAC observation,
``client`` talks to the server. lerobot side: ``batch`` builds the policy
frame, ``local_policy`` holds checkpoint + processors + tactile encoder,
``app`` is the FastAPI app. ``wire`` is the array encoding both sides share.
"""
