## Fix netlist generation and visualization
# Authored 11/07/2024

# Handmade checks to ensure netlist graph creation is correct

from helpers import *
from spice_net import *

import numpy as np
import networkx as nx
import tqdm
import json
import matplotlib.pyplot as plt

import PySpice

grid_graph = """0 4 {'weight': 0.5390508031418598}
0 12 {'weight': 0.6721514930979355}
0 1 {'weight': 0.5822107008573151}
0 3 {'weight': 0.5359065463975176}
1 5 {'weight': 0.4389238394711238}
1 13 {'weight': 0.6167152904533248}
1 2 {'weight': 0.45006976901015405}
2 6 {'weight': 0.8134184006256638}
2 14 {'weight': 0.8709302084008235}
2 3 {'weight': 0.40675321506062223}
3 7 {'weight': 0.7333800304661316}
3 15 {'weight': 0.5231159358023236}
4 8 {'weight': 0.5544356488751458}
4 5 {'weight': 0.8404773106341289}
4 7 {'weight': 0.15682884655830956}
5 9 {'weight': 0.1697034397612326}
5 6 {'weight': 0.11617471795226059}
6 10 {'weight': 0.7660958764383504}
6 7 {'weight': 0.7225254007598804}
7 11 {'weight': 0.7960097185974554}
8 12 {'weight': 0.8828946737862112}
8 9 {'weight': 0.739326851373379}
8 11 {'weight': 0.4691834898023455}
9 13 {'weight': 0.7244233410291644}
9 10 {'weight': 0.1946195406951466}
10 14 {'weight': 0.6119368170620191}
10 11 {'weight': 0.21468262992723713}
11 15 {'weight': 0.8557351336396671}
12 13 {'weight': 0.5174786574000574}
12 15 {'weight': 0.4317295519924189}
13 14 {'weight': 0.3116444896837016}
14 15 {'weight': 0.7193869515473733}
"""
node_cfg = (np.array([[5, 16], [7, 16], [13, 16], [15, 16]]), np.array([[10, 0]]))
