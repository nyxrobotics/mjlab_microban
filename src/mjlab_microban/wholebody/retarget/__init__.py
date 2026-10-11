"""Human motion (BVH, AMASS SMPL-X, PICO recordings on G1) retargeted to Microban with GMR's IK.

core.py retargets, skeleton.py / smplx.py / pico.py read the sources,
model.py builds the IK model, kinds.py names CMU's kinds and render.py draws
a result.  The data pipeline is src/mjlab_microban/scripts/wholebody_make_data.py
(docs/wholebody_data.md).
"""
