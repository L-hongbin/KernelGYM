// Trusted diagnostics, independent of submitted kernels. No fast-math or FMA:
// each intermediate is rounded like the existing in-place PyTorch operations.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_reduce.cuh>
#include <cfloat>
#include <cstdint>
#include <climits>

struct Stats {
  double sum, maximum, score[3];
  int64_t first, last, index[3];
  int bad, nan, inf, buckets[4];
  unsigned rows, columns;
};

__device__ void insert(Stats &s, double value, int64_t index) {
  for (int k = 0; k < 3; ++k) {
    if (value > s.score[k] || (value == s.score[k] && index < s.index[k])) {
      for (int j = 2; j > k; --j) { s.score[j] = s.score[j-1]; s.index[j] = s.index[j-1]; }
      s.score[k] = value; s.index[k] = index;
      return;
    }
  }
}
__device__ Stats empty_stats() {
  Stats s{};
  s.first = INT64_MAX; s.last = -1;
  for (int k=0;k<3;++k) { s.score[k]=-1; s.index[k]=INT64_MAX; }
  return s;
}
struct Merge {
  __device__ Stats operator()(Stats a, const Stats &b) const {
    a.sum += b.sum;
    a.maximum = (isnan(a.maximum) || isnan(b.maximum)) ? NAN : fmax(a.maximum, b.maximum);
    a.first = min(a.first,b.first); a.last = max(a.last,b.last);
    a.bad += b.bad; a.nan += b.nan; a.inf += b.inf;
    a.rows |= b.rows; a.columns |= b.columns;
    for (int k=0;k<4;++k) a.buckets[k] += b.buckets[k];
    for (int k=0;k<3;++k) insert(a,b.score[k],b.index[k]);
    return a;
  }
};

template<class T> __device__ double rounded(double v) { return double(T(v)); }
template<> __device__ double rounded<__half>(double v) { return double(__half2float(__double2half(v))); }
template<> __device__ double rounded<__nv_bfloat16>(double v) { return double(__bfloat162float(__double2bfloat16(v))); }
template<class T> __device__ double opmath(double v) {
  return rounded<T>(sizeof(T)==8 ? v : double(float(v)));
}
template<class T> __device__ double read_value(const T* p, int64_t i) { return double(p[i]); }
template<> __device__ double read_value(const __half* p, int64_t i) { return double(__half2float(p[i])); }
template<> __device__ double read_value(const __nv_bfloat16* p, int64_t i) { return double(__bfloat162float(p[i])); }

template<class T> __global__ void summarize(const T* ref, const T* candidate, double* result,
    int64_t rows, int64_t columns, int64_t tile_rows, int64_t tile_columns,
    double atol, double rtol, double finite_max) {
  using Reduction = cub::BlockReduce<Stats,256>;
  __shared__ typename Reduction::TempStorage storage;
  const int64_t tile = blockIdx.x;
  const int64_t tc = tile % tile_columns;
  const int64_t tr = (tile / tile_columns) % tile_rows;
  const int64_t prefix = tile / (tile_rows * tile_columns);
  Stats s = empty_stats();
  // PyTorch scalar operations on half/bfloat16 use float opmath, not half scalars.
  const double a = sizeof(T)==8 ? atol : double(float(atol));
  const double r = sizeof(T)==8 ? rtol : double(float(rtol));
  for (int cell=threadIdx.x;cell<1024;cell+=256) {
    int rr=cell/32, cc=cell%32;
    int64_t row=tr*32+rr, col=tc*32+cc;
    if (row>=rows || col>=columns) continue;
    int64_t i=(prefix*rows+row)*columns+col;
    double x=read_value(ref,i), y=read_value(candidate,i);
    double diff=fabs(opmath<T>(y-x));
    double tol=opmath<T>(opmath<T>(fabs(x)*r)+a);
    bool bad=!(diff<=tol);
    s.sum+=diff;
    s.maximum=(isnan(diff)||isnan(s.maximum)) ? NAN : fmax(s.maximum,diff);
    s.nan+=isnan(y); s.inf+=isinf(y); s.bad+=bad;
    if(bad) { s.first=min(s.first,i); s.last=max(s.last,i); s.rows|=1u<<rr; s.columns|=1u<<cc; }
    double e=opmath<T>(diff/tol);
    if(!isfinite(e)) e=finite_max;
    if(atol==0 && diff==0 && tol==0) e=0;
    // One interval classification replaces four full-tensor comparisons.
    if(e<=2) ++s.buckets[0];
    else if(e<=4) ++s.buckets[1];
    else if(e<=8) ++s.buckets[2];
    else if(e<=16) ++s.buckets[3];
    insert(s,e,i);
  }
  Stats v=Reduction(storage).Reduce(s,Merge{});
  if(threadIdx.x==0) {
    double* o=result+tile*19;
    o[0]=v.maximum; o[1]=v.sum; o[2]=v.bad; o[3]=v.nan; o[4]=v.inf;
    for(int k=0;k<4;++k) o[5+k]=v.buckets[k];
    o[9]=double(v.first); o[10]=double(v.last); o[11]=v.rows; o[12]=v.columns;
    for(int k=0;k<3;++k) {o[13+2*k]=v.score[k];o[14+2*k]=double(v.index[k]);}
  }
}

extern "C" int kg_correctness(const void* ref,const void* candidate,void* result,
    int dtype,int64_t rows,int64_t columns,int64_t prefix,double atol,double rtol,void* stream) {
  int64_t tr=(rows+31)/32, tc=(columns+31)/32;
  dim3 grid(prefix*tr*tc);
  cudaStream_t s=static_cast<cudaStream_t>(stream);
  #define LAUNCH(T,MAX) summarize<T><<<grid,256,0,s>>>(static_cast<const T*>(ref),static_cast<const T*>(candidate),static_cast<double*>(result),rows,columns,tr,tc,atol,rtol,MAX)
  switch(dtype) {
    case 0: LAUNCH(float,FLT_MAX); break;
    case 1: LAUNCH(double,DBL_MAX); break;
    case 2: LAUNCH(__half,65504.0); break;
    case 3: LAUNCH(__nv_bfloat16,3.3895313892515355e38); break;
    default: return int(cudaErrorInvalidValue);
  }
  return int(cudaGetLastError());
}
