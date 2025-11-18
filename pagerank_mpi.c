// pagerank_mpi.c
// Compile: mpicc -O3 -o pagerank_mpi pagerank_mpi.c
// Usage: mpirun -np <P> ./pagerank_mpi graph.edgelist [d=0.85] [eps=1e-6] [maxiter=100]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

typedef long long ll;
void die(const char *s){ perror(s); MPI_Abort(MPI_COMM_WORLD,1); }

int main(int argc, char **argv){
    MPI_Init(&argc,&argv);
    int rank, nprocs; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    if(argc < 2){
        if(rank==0) fprintf(stderr, "Usage: %s graph.edgelist [d=0.85] [eps=1e-6] [maxiter=100]\n", argv[0]);
        MPI_Finalize(); return 1;
    }
    const char *fname = argv[1];
    double d = (argc>2)? atof(argv[2]) : 0.85;
    double eps = (argc>3)? atof(argv[3]) : 1e-6;
    int maxiter = (argc>4)? atoi(argv[4]) : 100;

    int N=0;
    size_t nnz=0;

    // rank 0 reads and partitions columns
    int *global_col_ptr = NULL;
    int *global_row_idx = NULL;
    double *global_val = NULL;
    int *global_outdeg = NULL;
    int *global_dangling = NULL;
    int global_ndang = 0;

    if(rank==0){
        FILE *f = fopen(fname,"r"); if(!f) die("fopen");
        int src,dst;
        int maxnode=-1; size_t cnt=0;
        while(fscanf(f,"%d %d",&src,&dst)==2){ if(src>maxnode) maxnode=src; if(dst>maxnode) maxnode=dst; cnt++; }
        N = maxnode+1; nnz = cnt;
        rewind(f);
        int *coo_src = malloc(sizeof(int)*nnz);
        int *coo_dst = malloc(sizeof(int)*nnz);
        size_t i=0;
        while(fscanf(f,"%d %d",&src,&dst)==2){ coo_src[i]=src; coo_dst[i]=dst; i++; }
        fclose(f);

        global_outdeg = calloc(N,sizeof(int));
        for(size_t e=0;e<nnz;e++) global_outdeg[coo_src[e]]++;

        // build CSC
        global_col_ptr = malloc(sizeof(int)*(N+1));
        global_col_ptr[0]=0;
        for(int c=0;c<N;c++) global_col_ptr[c+1] = global_col_ptr[c] + global_outdeg[c];
        global_row_idx = malloc(sizeof(int)*nnz);
        global_val = malloc(sizeof(double)*nnz);
        int *cur = malloc(sizeof(int)*N);
        for(int c=0;c<N;c++) cur[c]=global_col_ptr[c];
        for(size_t e=0;e<nnz;e++){
            int c = coo_src[e];
            int r = coo_dst[e];
            int p = cur[c]++;
            global_row_idx[p] = r;
        }
        for(int c=0;c<N;c++){
            double v = (global_outdeg[c]>0)? 1.0/(double)global_outdeg[c] : 0.0;
            for(int p=global_col_ptr[c]; p<global_col_ptr[c+1]; p++) global_val[p] = v;
        }
        free(cur); free(coo_src); free(coo_dst);

        // compute dangling list (global)
        global_dangling = malloc(sizeof(int)*N);
        global_ndang=0;
        for(int i=0;i<N;i++) if(global_outdeg[i]==0) global_dangling[global_ndang++]=i;
    }

    // broadcast N and nnz and global_ndang
    MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&nnz,1,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
    MPI_Bcast(&global_ndang,1,MPI_INT,0,MPI_COMM_WORLD);

    if(rank!=0){
        global_dangling = malloc(sizeof(int)*global_ndang);
    }
    // broadcast dangling list
    MPI_Bcast(global_dangling, global_ndang, MPI_INT, 0, MPI_COMM_WORLD);

    // Partition columns into contiguous blocks (same logic on all ranks)
    int cols_per = N / nprocs;
    int rem = N % nprocs;
    int c_start = rank * cols_per + (rank < rem ? rank : rem);
    int c_end   = c_start + cols_per + (rank < rem ? 1 : 0); // [c_start, c_end)

    // rank 0 sends each rank its local CSC fragment (col_ptr_local, row_idx_local, val_local)
    // local_ncols = c_end - c_start
    int local_ncols = c_end - c_start;
    int *local_col_ptr = NULL;
    int *local_row_idx = NULL;
    double *local_val = NULL;
    int local_nnz = 0;

    if(rank==0){
        // for each rank prepare/send their data (including rank 0)
        for(int rnk=0;rnk<nprocs;rnk++){
            int rs = rnk * cols_per + (rnk < rem ? rnk : rem);
            int re = rs + cols_per + (rnk < rem ? 1 : 0);
            int ln = re - rs;
            int start_ptr = global_col_ptr[rs];
            int end_ptr = global_col_ptr[re];
            int lnnz = end_ptr - start_ptr;
            if(rnk==0){
                local_ncols = ln;
                local_nnz = lnnz;
                local_col_ptr = malloc(sizeof(int)*(local_ncols+1));
                for(int i=0;i<=local_ncols;i++) local_col_ptr[i] = global_col_ptr[rs + i] - start_ptr;
                local_row_idx = malloc(sizeof(int)*local_nnz);
                memcpy(local_row_idx, &global_row_idx[start_ptr], sizeof(int)*local_nnz);
                local_val = malloc(sizeof(double)*local_nnz);
                memcpy(local_val, &global_val[start_ptr], sizeof(double)*local_nnz);
            } else {
                // send metadata then arrays
                MPI_Send(&ln, 1, MPI_INT, rnk, 1, MPI_COMM_WORLD);
                MPI_Send(&lnnz, 1, MPI_INT, rnk, 2, MPI_COMM_WORLD); // send local nnz
                // send col_ptr normalized
                int *tmp_col = malloc(sizeof(int)*(ln+1));
                for(int i=0;i<=ln;i++) tmp_col[i] = global_col_ptr[rs + i] - start_ptr;
                MPI_Send(tmp_col, ln+1, MPI_INT, rnk, 3, MPI_COMM_WORLD);
                MPI_Send(&global_row_idx[start_ptr], lnnz, MPI_INT, rnk, 4, MPI_COMM_WORLD);
                MPI_Send(&global_val[start_ptr], lnnz, MPI_DOUBLE, rnk, 5, MPI_COMM_WORLD);
                free(tmp_col);
            }
        }
    } else {
        // receive from rank0
        MPI_Recv(&local_ncols, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&local_nnz, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_col_ptr = malloc(sizeof(int)*(local_ncols+1));
        MPI_Recv(local_col_ptr, local_ncols+1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_row_idx = malloc(sizeof(int)*local_nnz);
        MPI_Recv(local_row_idx, local_nnz, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_val = malloc(sizeof(double)*local_nnz);
        MPI_Recv(local_val, local_nnz, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // All ranks now have: N, global_dangling (list), local_col_ptr, local_row_idx, local_val
    // allocate vectors
    double *r = malloc(sizeof(double)*N);
    double *r_new = malloc(sizeof(double)*N);
    double *y_local = malloc(sizeof(double)*N);
    double *y_global = malloc(sizeof(double)*N);

    // init r uniform
    for(int i=0;i<N;i++) r[i] = 1.0/(double)N;

    double t0_all = MPI_Wtime();
    double spmv_total = 0.0;
    int iter;
    for(iter=0; iter<maxiter; ++iter){
        double t_spmv0 = MPI_Wtime();
        // zero y_local
        for(int i=0;i<N;i++) y_local[i]=0.0;

        // compute contributions from local columns (local_col_ptr is relative to local arrays)
        int ccount = local_ncols;
        int base_col = c_start;
        for(int lc=0; lc<ccount; ++lc){
            int col_global = base_col + lc;
            double rcol = r[col_global];
            if(rcol==0.0) continue;
            int ptr0 = local_col_ptr[lc];
            int ptr1 = local_col_ptr[lc+1];
            for(int p=ptr0; p<ptr1; p++){
                int row = local_row_idx[p];
                y_local[row] += local_val[p] * rcol;
            }
        }
        double t_spmv1 = MPI_Wtime();
        spmv_total += t_spmv1 - t_spmv0;

        // Allreduce y_local -> y_global
        MPI_Allreduce(y_local, y_global, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // compute dangling sum local (use global dangling list)
        double local_dang = 0.0;
        for(int i=0;i<global_ndang;i++) local_dang += r[ global_dangling[i] ];
        double dangling_sum = 0.0;
        MPI_Allreduce(&local_dang, &dangling_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // compute r_new and local diff
        double local_diff = 0.0;
        for(int i=0;i<N;i++){
            r_new[i] = d*( y_global[i] + dangling_sum / (double)N ) + (1.0-d)/(double)N;
            local_diff += fabs(r_new[i] - r[i]);
        }
        double global_diff = 0.0;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // swap
        double *tmp = r; r = r_new; r_new = tmp;

        if(global_diff < eps){ iter++; break; }
    }
    double t1_all = MPI_Wtime();
    double total_time = t1_all - t0_all;
    if(iter==maxiter) iter = maxiter;

    // metrics (collect on rank 0)
    double edges = (double) nnz;
    double mteps = (edges * (double)iter) / total_time / 1e6;
    double gflops = (2.0 * edges * (double)iter) / total_time / 1e9;

    if(rank==0){
        FILE *fr = fopen("pagerank_mpi.rvec","w");
        for(int i=0;i<N;i++) fprintf(fr, "%d %.12e\n", i, r[i]);
        fclose(fr);
        FILE *fs = fopen("pagerank_mpi.stats","w");
        fprintf(fs,"N %d\nnnz %zu\nprocs %d\niterations %d\ntotal_time %g\nspmv_time %g\nMTEPS %g\nGFLOP_s %g\n",
                N, nnz, nprocs, iter, total_time, spmv_total, mteps, gflops);
        fclose(fs);
        printf("MPI PageRank finished: N=%d nnz=%zu procs=%d iterations=%d total_time=%g spmv_time=%g MTEPS=%g GFLOP/s=%g\n",
               N, nnz, nprocs, iter, total_time, spmv_total, mteps, gflops);
    }

    if(local_col_ptr) free(local_col_ptr);
    if(local_row_idx) free(local_row_idx);
    if(local_val) free(local_val);
    if(global_col_ptr) free(global_col_ptr);
    if(global_row_idx) free(global_row_idx);
    if(global_val) free(global_val);
    if(global_outdeg) free(global_outdeg);
    if(global_dangling) free(global_dangling);
    free(r); free(r_new); free(y_local); free(y_global);
    MPI_Finalize();
    return 0;
}
