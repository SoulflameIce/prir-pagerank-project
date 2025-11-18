import random
import os

base = "/graphs"
os.makedirs(base, exist_ok=True)

graphs = {
    "graph1.txt": (10, 20),
    "graph2.txt": (50, 150),
    "graph3.txt": (200, 800),
    "graph4.txt": (1000, 5000),
    "graph5.txt": (5000, 30000),
    "graph6.txt": (30000, 60000),
    "graph7.txt": (60000, 100000)
}

for name, (nodes, edges) in graphs.items():
    path = os.path.join(base, name)
    with open(path, "w") as f:
        for _ in range(edges):
            src = random.randrange(nodes)
            dst = random.randrange(nodes)
            if src != dst:
                f.write(f"{src} {dst}\n")

list(os.listdir(base))
