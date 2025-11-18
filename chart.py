import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# LOAD NEW FORMAT (WITH graph column)
# ============================================

cuda = pd.read_csv("results5/pagerank_cuda_stats.csv")
mpi  = pd.read_csv("results5/pagerank_mpi_stats.csv")
omp  = pd.read_csv("results5/pagerank_openmp_stats.csv")

# Fill missing columns
cuda["threads"] = np.nan
cuda["procs"]   = np.nan

# ============================================
# SPEEDUP & EFFICIENCY PER GRAPH
# ============================================

def compute_speedup(df, parallel_col):
    results = {}

    for graph, group in df.groupby("graph"):
        group = group.copy()

        if parallel_col is None:
            # CUDA: speedup = 1
            group["speedup"] = 1
            group["efficiency"] = 1
        else:
            base = group[group[parallel_col] == group[parallel_col].min()]["total_time"].iloc[0]
            group["speedup"] = base / group["total_time"]
            group["efficiency"] = group["speedup"] / group[parallel_col]

        results[graph] = group

    return results

omp_results  = compute_speedup(omp, "threads")
mpi_results  = compute_speedup(mpi, "procs")
cuda_results = compute_speedup(cuda, None)

# ============================================
# PLOT SPEEDUP / EFFICIENCY
# ============================================

def plot_speedup(results, label):
    plt.figure(figsize=(8,5))
    for graph, df in results.items():
        xcol = "threads" if "threads" in df.columns and df["threads"].notna().any() else \
               "procs" if "procs" in df.columns and df["procs"].notna().any() else None

        if xcol is None:
            x = [1]  # CUDA single "thread"
            y = df["speedup"]
        else:
            x = df[xcol].dropna()
            y = df["speedup"].loc[x.index]

        plt.plot(x, y, marker="o", label=graph)

    plt.xlabel("Threads / Processes")
    plt.ylabel("Speedup")
    plt.title(f"Speedup – {label}")
    plt.legend()
    plt.grid()
    plt.show()


def plot_efficiency(results, label):
    plt.figure(figsize=(8,5))
    for graph, df in results.items():
        xcol = "threads" if "threads" in df.columns and df["threads"].notna().any() else \
               "procs" if "procs" in df.columns and df["procs"].notna().any() else None

        if xcol is None:
            x = [1]  # CUDA
            y = df["efficiency"]
        else:
            x = df[xcol].dropna()
            y = df["efficiency"].loc[x.index]

        plt.plot(x, y, marker="o", label=graph)

    plt.xlabel("Threads / Processes")
    plt.ylabel("Efficiency")
    plt.title(f"Efficiency – {label}")
    plt.legend()
    plt.grid()
    plt.show()

plot_speedup(omp_results, "OpenMP")
plot_efficiency(omp_results, "OpenMP")

plot_speedup(mpi_results, "MPI")
plot_efficiency(mpi_results, "MPI")

# ============================================
# BEST RESULTS PER GRAPH (CUDA vs MPI vs OMP)
# ============================================

best_cuda = cuda.loc[cuda.groupby("graph")["MTEPS"].idxmax()]
best_omp  = omp.loc[ omp.groupby("graph")["MTEPS"].idxmax() ]
best_mpi  = mpi.loc[ mpi.groupby("graph")["MTEPS"].idxmax() ]

plt.figure(figsize=(9,6))
plt.plot(best_cuda["graph"], best_cuda["MTEPS"], marker="o", label="CUDA")
plt.plot(best_omp["graph"],  best_omp["MTEPS"], marker="o", label="OpenMP")
plt.plot(best_mpi["graph"],  best_mpi["MTEPS"], marker="o", label="MPI")
plt.xlabel("Graph")
plt.ylabel("MTEPS")
plt.title("Best performance per graph")
plt.grid()
plt.legend()
plt.show()


# ============================================
# BEST GFLOPS COMPARISON (CUDA vs MPI vs OMP)
# ============================================

best_cuda_flops = cuda.loc[cuda.groupby("graph")["GFLOPS"].idxmax()]
best_omp_flops  = omp.loc[ omp.groupby("graph")["GFLOPS"].idxmax() ]
best_mpi_flops  = mpi.loc[ mpi.groupby("graph")["GFLOPS"].idxmax() ]

plt.figure(figsize=(9,6))
plt.plot(best_cuda_flops["graph"], best_cuda_flops["GFLOPS"], marker="o", label="CUDA")
plt.plot(best_omp_flops["graph"],  best_omp_flops["GFLOPS"], marker="o", label="OpenMP")
plt.plot(best_mpi_flops["graph"],  best_mpi_flops["GFLOPS"], marker="o", label="MPI")

plt.xlabel("Graph")
plt.ylabel("GFLOPS")
plt.title("Best GFLOPS per graph (CUDA vs MPI vs OpenMP)")
plt.grid()
plt.legend()
plt.show()


# ============================================
# SUMMARY TABLE (MERGED BY GRAPH)
# ============================================
