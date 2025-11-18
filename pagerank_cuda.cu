// pagerank_cuda.cu
// Compile: nvcc -O3 -arch=sm_60 -o pagerank_cuda pagerank_cuda.cu
// Usage: ./pagerank_cuda graph.edgelist [d=0.85] [eps=1e-6] [maxiter=100]
//
// Notes:
// - All reductions (dangling sum, L1 diff) are on-GPU using block reductions + atomicAdd.
// - atomicAdd(double) requires GPU arch supporting double atomicAdd (compute capability >= 6.0).
// - Input: edge-list (src dst), 0-based indices.

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s:%d: %s\n", __FILE__,__LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

__global__ void spmv_csr_kernel(int N, const int *row_ptr, const int *col_idx, const double *data, const double *r, double *y){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    double sum = 0.0;
    int start = row_ptr[i];
    int end   = row_ptr[i+1];
    for(int p = start; p < end; ++p){
        int col = col_idx[p];
        sum += data[p] * r[col];
    }
    y[i] = sum;
}

// add scalar to each entry: y[i] += val
__global__ void add_scalar_kernel(double *y, double val, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N) y[i] += val;
}

// update r_new = d*(y + dangling_contrib) + one_minus_d_over_N
__global__ void update_kernel(int N, double d, double dangling_contrib, double one_minus_d_over_N, const double *y, double *r_new){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N) r_new[i] = d * ( y[i] + dangling_contrib ) + one_minus_d_over_N;
}

// block-level reduction for arbitrary input array 'arr' where each thread supplies a value.
// Each block writes partial sum via atomicAdd to global_out (double*).
__global__ void block_reduce_sum_double(const double *arr, long len, double *global_out){
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    long gid = blockIdx.x * blockDim.x + tid;
    double x = 0.0;
    if(gid < len) x = arr[gid];
    sdata[tid] = x;
    __syncthreads();

    for(unsigned int s = blockDim.x >> 1; s>0; s >>= 1){
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if(tid == 0){
        atomicAdd(global_out, sdata[0]); // atomicAdd on double (compute capability >= 6.0)
    }
}

// compute abs diff per element into arr_diff, then do block reduction via same kernel.
// Here we provide a kernel that computes abs diff and reduces in-block directly.
__global__ void diff_reduce_kernel(const double *a, const double *b, long n, double *global_out){
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    long gid = blockIdx.x * blockDim.x + tid;
    double v = 0.0;
    if(gid < n) v = fabs(a[gid] - b[gid]);
    sdata[tid] = v;
    __syncthreads();

    for(unsigned int s = blockDim.x >> 1; s>0; s >>= 1){
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if(tid == 0){
        atomicAdd(global_out, sdata[0]);
    }
}

// sum dangling entries by indexing dangling_idx array
__global__ void dangling_reduce_kernel(const double *r, const int *dangling_idx, int ndang, double *global_out){
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    double x = 0.0;
    if(gid < ndang) x = r[ dangling_idx[gid] ];
    sdata[tid] = x;
    __syncthreads();

    for(unsigned int s = blockDim.x >> 1; s>0; s >>= 1){
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if(tid == 0){
        atomicAdd(global_out, sdata[0]);
    }
}

int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"Usage: %s graph.edgelist [d=0.85] [eps=1e-6] [maxiter=100]\n", argv[0]); return 1; }
    const char *fname = argv[1];
    double d = (argc>2)? atof(argv[2]) : 0.85;
    double eps = (argc>3)? atof(argv[3]) : 1e-6;
    int maxiter = (argc>4)? atoi(argv[4]) : 100;

    // read edge list: determine N, nnz
    FILE *f = fopen(fname,"r"); if(!f){ perror("fopen"); return 1; }
    int src,dst;
    int maxnode=-1;
    size_t nnz = 0;
    while(fscanf(f,"%d %d",&src,&dst)==2){ if(src>maxnode) maxnode=src; if(dst>maxnode) maxnode=dst; nnz++; }
    int N = maxnode+1;
    rewind(f);
    std::vector<int> coo_src(nnz), coo_dst(nnz);
    size_t idx=0;
    while(fscanf(f,"%d %d",&src,&dst)==2){ coo_src[idx]=src; coo_dst[idx]=dst; idx++; }
    fclose(f);

    // compute outdeg
    std::vector<int> outdeg(N,0);
    for(size_t e=0;e<nnz;e++) outdeg[ coo_src[e] ]++;

    // build CSR of matrix A where A[dst, src] = 1/outdeg[src] (rows = dst)
    std::vector<int> row_nnz(N,0);
    for(size_t e=0;e<nnz;e++) row_nnz[ coo_dst[e] ]++;
    std::vector<int> row_ptr(N+1,0);
    for(int i=0;i<N;i++) row_ptr[i+1] = row_ptr[i] + row_nnz[i];
    std::vector<int> cur = row_ptr;
    std::vector<int> col_idx(nnz);
    std::vector<double> data(nnz);
    for(size_t e=0;e<nnz;e++){
        int r = coo_dst[e];
        int c = coo_src[e];
        int p = cur[r]++;
        col_idx[p] = c;
        data[p] = (outdeg[c]>0) ? 1.0 / (double)outdeg[c] : 0.0;
    }

    // dangling list
    std::vector<int> dangling;
    for(int i=0;i<N;i++) if(outdeg[i]==0) dangling.push_back(i);
    int ndang = (int)dangling.size();

    // host vectors
    std::vector<double> h_r(N, 1.0/(double)N);

    // device allocations
    int *d_row_ptr = NULL, *d_col_idx = NULL, *d_dangling = NULL;
    double *d_data = NULL, *d_r = NULL, *d_y = NULL, *d_rnew = NULL;
    double *d_reduce_tmp = NULL; // used as single double for reductions (dangling sum, diff)
    CUDA_CHECK(cudaMalloc(&d_row_ptr, sizeof(int)*(N+1)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, sizeof(int)*nnz));
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(double)*nnz));
    CUDA_CHECK(cudaMalloc(&d_r, sizeof(double)*N));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(double)*N));
    CUDA_CHECK(cudaMalloc(&d_rnew, sizeof(double)*N));
    if(ndang>0) CUDA_CHECK(cudaMalloc(&d_dangling, sizeof(int)*ndang));
    CUDA_CHECK(cudaMalloc(&d_reduce_tmp, sizeof(double))); // single double for reductions

    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(int)*(N+1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), sizeof(int)*nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), sizeof(double)*nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), sizeof(double)*N, cudaMemcpyHostToDevice));
    if(ndang>0) CUDA_CHECK(cudaMemcpy(d_dangling, dangling.data(), sizeof(int)*ndang, cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (N + block - 1) / block;
    int grid_dang = (ndang + block - 1) / block;
    size_t shared_mem = block * sizeof(double);

    double t0 = (double)clock() / CLOCKS_PER_SEC;
    double spmv_total = 0.0;
    int iter;
    for(iter=0; iter<maxiter; ++iter){
        double t_spmv0 = (double)clock() / CLOCKS_PER_SEC;
        // SpMV: y = A * r
        spmv_csr_kernel<<<grid, block>>>(N, d_row_ptr, d_col_idx, d_data, d_r, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t_spmv1 = (double)clock() / CLOCKS_PER_SEC;
        spmv_total += (t_spmv1 - t_spmv0);

        // compute dangling sum on GPU (reduce r[dangling[i]])
        double dang_sum = 0.0;
        if(ndang>0){
            CUDA_CHECK(cudaMemset(d_reduce_tmp, 0, sizeof(double)));
            dangling_reduce_kernel<<<grid_dang, block, shared_mem>>>(d_r, d_dangling, ndang, d_reduce_tmp);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&dang_sum, d_reduce_tmp, sizeof(double), cudaMemcpyDeviceToHost));
        } else {
            dang_sum = 0.0;
        }
        double dangling_contrib = dang_sum / (double)N;

        // add dangling contribution to y: y[i] += dangling_contrib
        add_scalar_kernel<<<grid, block>>>(d_y, dangling_contrib, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // update r_new on GPU
        double one_minus_d_over_N = (1.0 - d) / (double)N;
        update_kernel<<<grid, block>>>(N, d, 0.0 /* dangling already added to y */, one_minus_d_over_N, d_y, d_rnew);
        CUDA_CHECK(cudaDeviceSynchronize());

        // compute L1 diff on GPU: diff = sum |r - rnew|
        CUDA_CHECK(cudaMemset(d_reduce_tmp, 0, sizeof(double)));
        diff_reduce_kernel<<<grid, block, shared_mem>>>(d_r, d_rnew, N, d_reduce_tmp);
        CUDA_CHECK(cudaDeviceSynchronize());
        double diff = 0.0;
        CUDA_CHECK(cudaMemcpy(&diff, d_reduce_tmp, sizeof(double), cudaMemcpyDeviceToHost));

        // swap r and rnew
        double *tmp = d_r; d_r = d_rnew; d_rnew = tmp;

        if(diff < eps){ iter++; break; }
    }
    double t1 = (double)clock() / CLOCKS_PER_SEC;
    double total_time = t1 - t0;
    if(iter==maxiter) iter = maxiter;

    // copy final r back
    CUDA_CHECK(cudaMemcpy(h_r.data(), d_r, sizeof(double)*N, cudaMemcpyDeviceToHost));

    double edges = (double) nnz;
    double mteps = (edges * (double)iter) / total_time / 1e6;
    double gflops = (2.0 * edges * (double)iter) / total_time / 1e9;

    // write outputs
    FILE *fr = fopen("pagerank_cuda.rvec","w");
    for(int i=0;i<N;i++) fprintf(fr, "%d %.12e\n", i, h_r[i]);
    fclose(fr);
    FILE *fs = fopen("pagerank_cuda.stats","w");
    fprintf(fs,"N %d\nnnz %zu\niterations %d\ntotal_time %g\nspmv_time %g\nMTEPS %g\nGFLOP_s %g\n", N, nnz, iter, total_time, spmv_total, mteps, gflops);
    fclose(fs);

    printf("CUDA PageRank finished: N=%d nnz=%zu iterations=%d total_time=%g spmv_time=%g MTEPS=%g GFLOP/s=%g\n",
           N, nnz, iter, total_time, spmv_total, mteps, gflops);

    CUDA_CHECK(cudaFree(d_row_ptr)); CUDA_CHECK(cudaFree(d_col_idx)); CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_r)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_rnew));
    if(ndang>0) CUDA_CHECK(cudaFree(d_dangling));
    CUDA_CHECK(cudaFree(d_reduce_tmp));
    return 0;
}
