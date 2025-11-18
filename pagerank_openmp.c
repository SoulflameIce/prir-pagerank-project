// pagerank_openmp.c
// Compile: gcc -O3 -fopenmp -march=native -o pagerank_openmp pagerank_openmp.c
// Usage: ./pagerank_openmp graph.edgelist [d=0.85] [eps=1e-6] [maxiter=100] [threads=8]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

typedef long long ll;
void die(const char *s){ perror(s); exit(1); }

int main(int argc, char **argv){
    if(argc < 2){
        fprintf(stderr, "Usage: %s graph.edgelist [d=0.85] [eps=1e-6] [maxiter=100] [threads=8]\n", argv[0]);
        return 1;
    }
    const char *fname = argv[1];
    double d = (argc>2)? atof(argv[2]) : 0.85;
    double eps = (argc>3)? atof(argv[3]) : 1e-6;
    int maxiter = (argc>4)? atoi(argv[4]) : 100;
    int nthreads = (argc>5)? atoi(argv[5]) : 8;
    omp_set_num_threads(nthreads);

    // Read edge list, determine N and nnz
    FILE *f = fopen(fname,"r");
    if(!f) die("fopen");
    int src, dst;
    int maxnode = -1;
    size_t nnz = 0;
    while(fscanf(f,"%d %d",&src,&dst)==2){
        if(src>maxnode) maxnode = src;
        if(dst>maxnode) maxnode = dst;
        nnz++;
    }
    int N = maxnode + 1;
    rewind(f);

    // Read into COO arrays
    int *coo_src = malloc(sizeof(int)*nnz);
    int *coo_dst = malloc(sizeof(int)*nnz);
    if(!coo_src||!coo_dst) die("malloc");
    size_t idx=0;
    while(fscanf(f,"%d %d",&src,&dst)==2){
        coo_src[idx]=src;
        coo_dst[idx]=dst;
        idx++;
    }
    fclose(f);

    // Compute outdegree per column (src)
    int *outdeg = calloc(N,sizeof(int));
    for(size_t i=0;i<nnz;i++) outdeg[coo_src[i]]++;
    // Build CSC: for each column (src) -> col_ptr
    int *col_ptr = malloc(sizeof(int)*(N+1));
    col_ptr[0]=0;
    for(int c=0;c<N;c++) col_ptr[c+1] = col_ptr[c] + outdeg[c];
    int *row_idx = malloc(sizeof(int)*nnz);
    double *val = malloc(sizeof(double)*nnz);
    if(!col_ptr || !row_idx || !val) die("malloc2");
    // temporary counter = copy of col_ptr
    int *cur = malloc(sizeof(int)*N);
    for(int c=0;c<N;c++) cur[c]=col_ptr[c];
    for(size_t e=0;e<nnz;e++){
        int c = coo_src[e];
        int r = coo_dst[e];
        int p = cur[c]++;
        row_idx[p] = r;
    }
    // fill val = 1/outdeg[col] for each column entry
    for(int c=0;c<N;c++){
        double v = (outdeg[c]>0)? 1.0 / (double)outdeg[c] : 0.0;
        for(int p=col_ptr[c]; p<col_ptr[c+1]; p++) val[p] = v;
    }
    free(cur);
    free(coo_src); free(coo_dst);

    // dangling nodes list
    int *dangling = malloc(sizeof(int)*N);
    int ndang=0;
    for(int i=0;i<N;i++) if(outdeg[i]==0) dangling[ndang++] = i;

    // allocate vectors
    double *r = malloc(sizeof(double)*N);
    double *r_new = malloc(sizeof(double)*N);
    double *y = malloc(sizeof(double)*N);
    for(int i=0;i<N;i++) r[i] = 1.0 / (double)N;

    double t_start = omp_get_wtime();
    double spmv_total = 0.0;
    int iter;
    for(iter=0; iter<maxiter; ++iter){
        double t0 = omp_get_wtime();
        // zero y
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N;i++) y[i]=0.0;

        // SpMV: y = P * r  (P stored CSC)
        #pragma omp parallel for schedule(dynamic,64)
        for(int c=0;c<N;c++){
            double rcol = r[c];
            if(rcol==0.0) continue;
            for(int p=col_ptr[c]; p<col_ptr[c+1]; p++){
                int row = row_idx[p];
                double add = val[p] * rcol;
                #pragma omp atomic
                y[row] += add;
            }
        }
        double t1 = omp_get_wtime();
        spmv_total += (t1 - t0);

        // dangling sum
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum)
        for(int i=0;i<ndang;i++) {
            int j = dangling[i];
            dangling_sum += r[j];
        }

        // update r_new and compute L1 diff
        double diff = 0.0;
        #pragma omp parallel for reduction(+:diff)
        for(int i=0;i<N;i++){
            r_new[i] = d*( y[i] + dangling_sum / (double)N ) + (1.0-d)/(double)N;
            diff += fabs(r_new[i] - r[i]);
        }

        // swap
        double *tmp = r; r = r_new; r_new = tmp;

        if(diff < eps) { iter++; break; } // iter is number of iterations performed
    }
    double t_end = omp_get_wtime();
    double total_time = t_end - t_start;
    if(iter==maxiter) iter = maxiter;

    // metrics
    double edges = (double) nnz;
    double mteps = (edges * (double)iter) / total_time / 1e6;
    double gflops = (2.0 * edges * (double)iter) / total_time / 1e9;

    // write pagerank vector
    FILE *fr = fopen("pagerank_openmp.rvec","w");
    for(int i=0;i<N;i++) fprintf(fr, "%d %.12e\n", i, r[i]);
    fclose(fr);

    // write stats
    FILE *fs = fopen("pagerank_openmp.stats","w");
    fprintf(fs,"N %d\nnnz %zu\nthreads %d\niterations %d\ntotal_time %g\nspmv_time %g\nMTEPS %g\nGFLOP_s %g\n", N, nnz, nthreads, iter, total_time, spmv_total, mteps, gflops);
    fclose(fs);

    // print brief
    printf("OpenMP PageRank finished: N=%d nnz=%zu threads=%d iterations=%d total_time=%g spmv_time=%g MTEPS=%g GFLOP/s=%g\n",
           N, nnz, nthreads, iter, total_time, spmv_total, mteps, gflops);

    free(col_ptr); free(row_idx); free(val); free(outdeg);
    free(dangling); free(r); free(r_new); free(y);
    return 0;
}
