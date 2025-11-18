#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# konfiguracja
# -----------------------------
GRAPH_DIR = "graphs"
OUTPUT_DIR = "results"
CSV_FILE = f"{OUTPUT_DIR}/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

PROGRAMS = {
    "openmp": "./pagerank_openmp",
    "mpi": "mpirun -np 8 ./pagerank_mpi",
    "cuda": "./pagerank_cuda"
}

PLOT_STYLES = {
    "openmp": "o-",
    "mpi": "s-",
    "cuda": "^-"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# pomocnicze funkcje
# -----------------------------

def parse_output(text: str) -> dict:
    """
    Oczekiwany format stdout:
        iterations: X
        time: Y
        mteps: Z
        gflops: W
    """
    result = {}
    for line in text.splitlines():
        line = line.strip().lower()
        if line.startswith("iterations"):
            result["iterations"] = int(line.split(":")[1])
        elif line.startswith("time"):
            result["time"] = float(line.split(":")[1])
        elif line.startswith("mteps"):
            result["mteps"] = float(line.split(":")[1])
        elif line.startswith("gflops"):
            result["gflops"] = float(line.split(":")[1])
    return result


def run_program(cmd):
    print(f"? uruchamiam: {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = result.stdout.decode("utf-8", errors="ignore")
    print(text)
    return parse_output(text)


# -----------------------------
# główna petla benchmarkowa
# -----------------------------

rows = []

graphs = sorted(os.listdir(GRAPH_DIR))
graphs = [g for g in graphs if g.endswith(".txt")]

print("\nRozpoczynam benchmark...\n")

for g in graphs:
    graph_path = os.path.join(GRAPH_DIR, g)
    print(f"\n=============================")
    print(f"    Graf: {graph_path}")
    print(f"=============================\n")

    # policz liczbe krawedzi
    with open(graph_path) as f:
        edges = sum(1 for _ in f)

    for name, program in PROGRAMS.items():
        cmd = f"{program} {graph_path}"

        try:
            out = run_program(cmd)
            out["program"] = name
            out["graph"] = g
            out["edges"] = edges
            rows.append(out)
        except Exception as e:
            print(f"Blad uruchamiania {name}: {e}")


# -----------------------------
# zapis CSV
# -----------------------------

df = pd.DataFrame(rows)
df.to_csv(CSV_FILE, index=False)
print(f"\nWyniki zapisano do: {CSV_FILE}\n")


# -----------------------------
# generowanie wykresów
# -----------------------------

def plot_metric(metric, ylabel):
    plt.figure(figsize=(10, 6))

    for prog in PROGRAMS.keys():
        subset = df[df["program"] == prog]
        subset = subset.sort_values("edges")

        plt.plot(
            subset["edges"],
            subset[metric],
            PLOT_STYLES[prog],
            label=prog.upper(),
            linewidth=2
        )

    plt.xlabel("Liczba krawedzi")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs rozmiar grafu")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/{metric}.png"
    plt.savefig(out, dpi=200)
    print(f"?? zapisano wykres: {out}")


plot_metric("time", "Czas wykonania [s]")
plot_metric("iterations", "Liczba iteracji")
plot_metric("mteps", "MTEPS")
plot_metric("gflops", "GFLOP/s")

print("\nBenchmark zakonczony!\n")
