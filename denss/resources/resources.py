import json
import os

path = os.path.dirname(__file__)

with open(os.path.join(path, "electrons.json"), 'r') as f:
    electrons = json.load(f)

with open(os.path.join(path, "mean_volumes.json"), 'r') as f:
    atomic_volumes = json.load(f)

with open(os.path.join(path, "numH.json"), 'r') as f:
    numH = json.load(f)

with open(os.path.join(path, "volH.json"), 'r') as f:
    volH = json.load(f)

with open(os.path.join(path, "vdW.json"), 'r') as f:
    vdW = json.load(f)

with open(os.path.join(path, "radii_scale_factors.json"), 'r') as f:
    radii_sf_dict = json.load(f)

with open(os.path.join(path, "cromer_mann_4gauss.json"), 'r') as f:
    ffcoeff = json.load(f)

