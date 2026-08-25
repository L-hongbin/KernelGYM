import concurrent.futures
import json
import time
import urllib.request
import uuid

URL = "http://127.0.0.1:20111/evaluate"


def ref(body, inputs):
    return f"""import torch
from torch import nn
class Model(nn.Module):
{body}
def get_init_inputs(): return []
def get_inputs():
{inputs}
"""


PREFIX = """import torch
from torch import nn
import triton
import triton.language as tl
"""


CASES = [
    (
        "add_odd_fp32",
        ref("    def forward(self, x): return x + 1.25", "    return [torch.randn(100003)]"),
        PREFIX
        + r"""
@triton.jit
def add_kernel(x, y, n: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(y + i, tl.load(x + i, mask=m) + 1.25, mask=m)
class ModelNew(nn.Module):
    def forward(self, x):
        y=torch.empty_like(x); add_kernel[(triton.cdiv(x.numel(),256),)](x,y,x.numel(),BLOCK=256); return y
""",
        "fp32",
        "auto",
    ),
    (
        "silu_gate_bf16",
        ref(
            "    def forward(self, x):\n        a,b=x.chunk(2,-1); return torch.nn.functional.silu(a)*b",
            "    return [torch.randn(257,8192,dtype=torch.bfloat16)]",
        ),
        PREFIX
        + r"""
@triton.jit
def silu_gate_kernel(x,y,n:tl.constexpr,H:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=i<n; row=i//H; col=i%H; base=row*(2*H)+col; a=tl.load(x+base,mask=m).to(tl.float32); b=tl.load(x+base+H,mask=m).to(tl.float32)
    tl.store(y+i,(a*tl.sigmoid(a)*b),mask=m)
class ModelNew(nn.Module):
    def forward(self,x):
        n=x.numel()//2; y=torch.empty(x.shape[0],x.shape[1]//2,device=x.device,dtype=x.dtype); silu_gate_kernel[(triton.cdiv(n,256),)](x,y,n,H=4096,BLOCK=256); return y
""",
        "bf16",
        "triton",
    ),
    (
        "rmsnorm_fp16",
        ref(
            "    def forward(self, x, w):\n        return x * torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6).to(x.dtype) * w",
            "    return [torch.randn(128,4096,dtype=torch.float16), torch.randn(4096,dtype=torch.float16)]",
        ),
        PREFIX
        + r"""
@triton.jit
def rms_kernel(x,w,y,H:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0); i=tl.arange(0,BLOCK); m=i<H; v=tl.load(x+row*H+i,mask=m,other=0.).to(tl.float32)
    inv=tl.rsqrt(tl.sum(v*v,axis=0)/H+EPS); ww=tl.load(w+i,mask=m,other=0.).to(tl.float32); tl.store(y+row*H+i,v*inv*ww,mask=m)
class ModelNew(nn.Module):
    def forward(self,x,w):
        y=torch.empty_like(x); rms_kernel[(x.shape[0],)](x,w,y,H=4096,EPS=1e-6,BLOCK=4096,num_warps=8); return y
""",
        "fp16",
        "triton",
    ),
    (
        "softmax_odd_fp32",
        ref("    def forward(self, x): return torch.softmax(x,dim=-1)", "    return [torch.randn(63,1000)]"),
        PREFIX
        + r"""
@triton.jit
def softmax_kernel(x,y,N:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0); i=tl.arange(0,BLOCK); m=i<N; v=tl.load(x+r*N+i,mask=m,other=-float("inf")); v=v-tl.max(v,axis=0); e=tl.exp(v); tl.store(y+r*N+i,e/tl.sum(e,axis=0),mask=m)
class ModelNew(nn.Module):
    def forward(self,x):
        y=torch.empty_like(x); softmax_kernel[(x.shape[0],)](x,y,N=1000,BLOCK=1024,num_warps=8); return y
""",
        "fp32",
        "triton",
    ),
    (
        "softcap_fp16",
        ref(
            "    def forward(self, x): return 30.0*torch.tanh(x/30.0)",
            "    return [torch.randn(131071,dtype=torch.float16)*50]",
        ),
        PREFIX
        + r"""
@triton.jit
def softcap_kernel(x,y,n:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=i<n; v=tl.load(x+i,mask=m).to(tl.float32); tl.store(y+i,30.*(2.*tl.sigmoid(2.*v/30.)-1.),mask=m)
class ModelNew(nn.Module):
    def forward(self,x):
        y=torch.empty_like(x); softcap_kernel[(triton.cdiv(x.numel(),256),)](x,y,x.numel(),BLOCK=256); return y
""",
        "fp16",
        "triton",
    ),
    (
        "rope_fp32",
        ref(
            "    def forward(self, x, c, s):\n        h=x.shape[-1]//2; return torch.cat((x[...,:h]*c-x[...,h:]*s,x[...,h:]*c+x[...,:h]*s),-1)",
            "    \n    x=torch.randn(37,8,128); a=torch.randn(37,8,64); return [x,torch.cos(a),torch.sin(a)]",
        ),
        PREFIX
        + r"""
@triton.jit
def rope_kernel(x,c,s,y,n:tl.constexpr,H:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=i<n; row=i//H; col=i%H; base=row*(2*H)+col
    a=tl.load(x+base,mask=m); b=tl.load(x+base+H,mask=m); cv=tl.load(c+i,mask=m); sv=tl.load(s+i,mask=m)
    tl.store(y+base,a*cv-b*sv,mask=m); tl.store(y+base+H,b*cv+a*sv,mask=m)
class ModelNew(nn.Module):
    def forward(self,x,c,s):
        y=torch.empty_like(x); n=c.numel(); rope_kernel[(triton.cdiv(n,256),)](x,c,s,y,n,H=64,BLOCK=256); return y
""",
        "fp32",
        "triton",
    ),
    (
        "kv_gather_bf16",
        ref(
            "    def forward(self, cache, slots): return cache[slots]",
            "    return [torch.randn(4096,8,128,dtype=torch.bfloat16), torch.randint(0,4096,(257,),dtype=torch.int64)]",
        ),
        PREFIX
        + r"""
@triton.jit
def gather_kernel(cache,slots,out,T:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    t=tl.program_id(0); i=tl.arange(0,BLOCK); m=i<D; src=tl.load(slots+t); v=tl.load(cache+src*D+i,mask=m); tl.store(out+t*D+i,v,mask=m)
class ModelNew(nn.Module):
    def forward(self,cache,slots):
        out=torch.empty((slots.numel(),)+cache.shape[1:],device=cache.device,dtype=cache.dtype); gather_kernel[(slots.numel(),)](cache,slots,out,T=257,D=1024,BLOCK=1024,num_warps=8); return out
""",
        "bf16",
        "triton",
    ),
    (
        "moe_weight_fp16",
        ref(
            "    def forward(self, x, expert, weight): return x*weight[:,None]",
            "    return [torch.randn(513,256,dtype=torch.float16), torch.randn(1024,256,dtype=torch.float16), torch.rand(513,dtype=torch.float16)]",
        ),
        PREFIX
        + r"""
@triton.jit
def moe_weight_kernel(x,w,y,n:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=i<n; row=i//D; v=tl.load(x+i,mask=m); ww=tl.load(w+row,mask=m); tl.store(y+i,v*ww,mask=m)
class ModelNew(nn.Module):
    def forward(self,x,expert,weight):
        y=torch.empty_like(x); moe_weight_kernel[(triton.cdiv(x.numel(),256),)](x,weight,y,x.numel(),D=256,BLOCK=256); return y
""",
        "fp16",
        "triton",
    ),
    (
        "matmul_fp16",
        ref(
            "    def forward(self, a, b): return a@b",
            "    return [torch.randn(128,256,dtype=torch.float16),torch.randn(256,192,dtype=torch.float16)]",
        ),
        PREFIX
        + r"""
@triton.jit
def mm_kernel(a,b,c,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1); mi=pm*BM+tl.arange(0,BM); ni=pn*BN+tl.arange(0,BN); ki=tl.arange(0,BK); acc=tl.zeros((BM,BN),tl.float32)
    for k in range(0,K,BK):
        av=tl.load(a+mi[:,None]*K+(k+ki[None,:]),mask=(mi[:,None]<M)&((k+ki[None,:])<K),other=0.)
        bv=tl.load(b+(k+ki[:,None])*N+ni[None,:],mask=((k+ki[:,None])<K)&(ni[None,:]<N),other=0.); acc+=tl.dot(av,bv)
    tl.store(c+mi[:,None]*N+ni[None,:],acc,mask=(mi[:,None]<M)&(ni[None,:]<N))
class ModelNew(nn.Module):
    def forward(self,a,b):
        c=torch.empty((a.shape[0],b.shape[1]),device=a.device,dtype=a.dtype); mm_kernel[(triton.cdiv(a.shape[0],32),triton.cdiv(b.shape[1],32))](a,b,c,M=128,N=192,K=256,BM=32,BN=32,BK=32,num_warps=4); return c
""",
        "fp16",
        "triton",
    ),
    (
        "noncontig_stride_fp32",
        ref("    def forward(self, x): return x*2.0-0.5", "    return [torch.randn(513,257).t()]"),
        PREFIX
        + r"""
@triton.jit
def stride_kernel(x,y,R:tl.constexpr,C:tl.constexpr,s0:tl.constexpr,s1:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=i<R*C; r=i//C; c=i%C; v=tl.load(x+r*s0+c*s1,mask=m); tl.store(y+i,v*2.-.5,mask=m)
class ModelNew(nn.Module):
    def forward(self,x):
        y=torch.empty(x.shape,device=x.device,dtype=x.dtype); stride_kernel[(triton.cdiv(x.numel(),256),)](x,y,R=257,C=513,s0=x.stride(0),s1=x.stride(1),BLOCK=256); return y
""",
        "fp32",
        "triton",
    ),
    (
        "causal_softmax_fp32",
        ref(
            "    def forward(self, x):\n        n=x.shape[-1]; p=torch.arange(n,device=x.device); return torch.softmax(x.masked_fill(p[None,None,:]>p[None,:,None],float('-inf')),dim=-1)",
            "    return [torch.randn(3,129,129)]",
        ),
        PREFIX
        + r"""
@triton.jit
def causal_softmax_kernel(x,y,N:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0); col=tl.arange(0,BLOCK); q=r%N; m=(col<N)&(col<=q); v=tl.load(x+r*N+col,mask=m,other=-float("inf")); v=v-tl.max(v,axis=0); e=tl.exp(v); tl.store(y+r*N+col,e/tl.sum(e,axis=0),mask=col<N)
class ModelNew(nn.Module):
    def forward(self,x):
        y=torch.empty_like(x); causal_softmax_kernel[(x.shape[0]*x.shape[1],)](x,y,N=129,BLOCK=256,num_warps=4); return y
""",
        "fp32",
        "triton",
    ),
    (
        "attention_merge_fp32",
        ref(
            "    def forward(self, v1, s1, v2, s2):\n        m=torch.maximum(s1,s2); a=torch.exp(s1-m); b=torch.exp(s2-m); return (v1*a[...,None]+v2*b[...,None])/(a+b)[...,None]",
            "    return [torch.randn(17,8,128),torch.randn(17,8),torch.randn(17,8,128),torch.randn(17,8)]",
        ),
        PREFIX
        + r"""
@triton.jit
def merge_kernel(v1,s1,v2,s2,y,N:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0); i=tl.arange(0,BLOCK); m=i<D; x=tl.load(s1+row); z=tl.load(s2+row); hi=tl.maximum(x,z); a=tl.exp(x-hi); b=tl.exp(z-hi); p=tl.load(v1+row*D+i,mask=m); q=tl.load(v2+row*D+i,mask=m); tl.store(y+row*D+i,(p*a+q*b)/(a+b),mask=m)
class ModelNew(nn.Module):
    def forward(self,v1,s1,v2,s2):
        y=torch.empty_like(v1); merge_kernel[(s1.numel(),)](v1,s1,v2,s2,y,N=136,D=128,BLOCK=128); return y
""",
        "fp32",
        "triton",
    ),
    (
        "torch_only_rejected",
        ref("    def forward(self, x): return x+1", "    return [torch.randn(1024)]"),
        "import torch\nfrom torch import nn\nclass ModelNew(nn.Module):\n    def forward(self,x): return x+1\n",
        "fp32",
        "triton",
    ),
]


def run(case, split=False):
    name, reference_code, kernel_code, precision, backend = case
    payload = {
        "task_id": f"wide-triton-{name}-{uuid.uuid4().hex[:8]}",
        "reference_code": reference_code,
        "kernel_code": kernel_code,
        "backend": backend,
        "precision": precision,
        "num_correct_trials": 3,
        "num_perf_trials": 12,
        "num_warmup": 3,
        "perf_trim_count": 1,
        "timeout": 300,
        "force_refresh": True,
        "enable_profiling": True,
        "detect_decoy_kernel": True,
        "split_compile_and_execute": split,
        "enable_compile_artifact_cache": split,
        "compiler_options": {"target": "cuda"} if split else None,
    }
    started = time.time()
    request = urllib.request.Request(URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=360) as response:
            result = json.load(response)
    except Exception as exc:
        return {"name": name, "transport_error": repr(exc), "wall_s": round(time.time() - started, 2)}
    metadata = result.get("metadata") or {}
    profiling = metadata.get("profiling") or {}
    return {
        "name": name,
        "backend": backend,
        "status": result.get("status"),
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "decoy": result.get("decoy_kernel"),
        "runtime_ms": result.get("kernel_runtime"),
        "speedup": result.get("speedup"),
        "profile_kernels": profiling.get("kernel_count"),
        "precompiled": metadata.get("precompiled_artifact_used"),
        "error": result.get("error_message"),
        "wall_s": round(time.time() - started, 2),
    }


if __name__ == "__main__":
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run, c): c[0] for c in CASES}
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    results.sort(key=lambda x: x["name"])
    print("RESULTS_JSON=" + json.dumps(results, ensure_ascii=False))
