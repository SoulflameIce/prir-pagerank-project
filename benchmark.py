import subprocess
import csv
import os

# ===============================
# KONFIGURACJA
# ===============================

GRAPHS = [f"graphs/graph{i}.txt" for i in range(1, 8)]
D = 0.85
EPS = 1e-6
MAXITER = 100

THREADS = [1, 2, 4, 8]
PROCS = [1, 2, 4, 8]

OUT_OPENMP = "results/pagerank_openmp_stats.csv"
OUT_MPI = "results/pagerank_mpi_stats.csv"
OUT_CUDA = "results/pagerank_cuda_stats.csv"

os.makedirs("results", exist_ok=True)


# ===============================
# FUNKCJA DO URUCHAMIANIA CMD
# ===============================

def run_cmd(cmd):
    print(f"\nRUNNING: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
    return result.stdout


# ===============================
# PARSER STATYSTYK
# ===============================

def parse_stats_output(output, extra_fields: dict):
    stats = {
        "graph": extra_fields.get("graph"),
        "threads": extra_fields.get("threads"),
        "procs": extra_fields.get("procs"),
        "N": None,
        "nnz": None,
        "iterations": None,
        "total_time": None,
        "spmv_time": None,
        "MTEPS": None,
        "GFLOPS": None
    }

    for line in output.splitlines():
        if "PageRank finished" in line:
            parts = line.replace(",", "").replace("=", " ").split()
            for i, p in enumerate(parts):
                if p == "N":
                    stats["N"] = int(parts[i+1])
                elif p == "nnz":
                    stats["nnz"] = int(parts[i+1])
                elif p == "iterations":
                    stats["iterations"] = int(parts[i+1])
                elif p == "total_time":
                    stats["total_time"] = float(parts[i+1])
                elif p == "spmv_time":
                    stats["spmv_time"] = float(parts[i+1])
                elif p in ("MTEPS", "MTEPS/s", "MTEPS/s:"):
                    stats["MTEPS"] = float(parts[i+1])
                elif p in ("GFLOP/s", "GFLOPS", "GFLOP_s"):
                    stats["GFLOPS"] = float(parts[i+1])

    return stats


# ===============================
# URUCHAMIANIE OPENMP
# ===============================

with open(OUT_OPENMP, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "graph", "threads", "N", "nnz", "iterations",
        "total_time", "spmv_time", "MTEPS", "GFLOPS"
    ])
    writer.writeheader()

    for graph in GRAPHS:
        for t in THREADS:

            cmd = [
                "./pagerank_openmp",
                graph,
                str(D),
                str(EPS),
                str(MAXITER),
                str(t)
            ]

            output = run_cmd(cmd)

            stats = parse_stats_output(output, {
                "graph": graph,
                "threads": t
            })

            # remove unused key
            stats.pop("procs", None)

            writer.writerow(stats)

print("\n=== OpenMP DONE ===")


# ===============================
# URUCHAMIANIE MPI
# ===============================

with open(OUT_MPI, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "graph", "procs", "N", "nnz", "iterations",
        "total_time", "spmv_time", "MTEPS", "GFLOPS"
    ])
    writer.writeheader()

    for graph in GRAPHS:
        for p in PROCS:

            cmd = [
                "mpirun",
                "-np", str(p),
                "./pagerank_mpi",
                graph,
                str(D),
                str(EPS),
                str(MAXITER)
            ]

            output = run_cmd(cmd)

            stats = parse_stats_output(output, {
                "graph": graph,
                "procs": p
            })

            # remove unused key
            stats.pop("threads", None)

            writer.writerow(stats)


print("\n=== MPI DONE ===")


# ===============================
# URUCHAMIANIE CUDA
# ===============================

with open(OUT_CUDA, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "graph", "N", "nnz", "iterations",
        "total_time", "spmv_time", "MTEPS", "GFLOPS"
    ])
    writer.writeheader()

    for graph in GRAPHS:

        cmd = [
            "./pagerank_cuda",
            graph,
            str(D),
            str(EPS),
            str(MAXITER)
        ]

        output = run_cmd(cmd)

        stats = parse_stats_output(output, {
            "graph": graph
        })

        # usuwamy nieużywane pola threads/procs
        stats.pop("threads")
        stats.pop("procs")

        writer.writerow(stats)

print("\n=== CUDA DONE ===")
print(f"\nWyniki zapisano do:\n- {OUT_OPENMP}\n- {OUT_MPI}\n- {OUT_CUDA}")
