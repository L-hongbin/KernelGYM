import concurrent.futures
import json
import time
import urllib.request
import uuid

URL = "http://127.0.0.1:20111/evaluate"
PREFIX = """import torch
from torch import nn
import tilelang
from tilelang import language as T
"""


def ref(body, inputs):
    return f"""import torch
from torch import nn
class Model(nn.Module):
{body}
def get_init_inputs(): return []
def get_inputs():
{inputs}
"""


def wrap(factory, args):
    return (
        PREFIX
        + factory
        + f"\ncompiled_kernel = make_kernel({args})\nclass ModelNew(nn.Module):\n    def forward(self,*args): return compiled_kernel(*args)\n"
    )


CASES = [
    (
        "add_odd_fp32",
        ref("    def forward(self,x): return x+1.25", "    return [torch.randn(100003)]"),
        wrap(
            r"""
@tilelang.jit(out_idx=[-1], target="cuda")
def make_kernel(N):
    @T.prim_func
    def add(A:T.Tensor((N,),"float32"),B:T.Tensor((N,),"float32")):
        with T.Kernel(T.ceildiv(N,256),threads=256) as bx:
            for tx in T.Parallel(256):
                i=bx*256+tx
                if i<N: B[i]=A[i]+T.float32(1.25)
    return add
""",
            "100003",
        ),
        "fp32",
        "auto",
    ),
    (
        "binary_bf16",
        ref(
            "    def forward(self,x,y): return x*0.75+y*1.5",
            "    return [torch.randn(257,513,dtype=torch.bfloat16),torch.randn(257,513,dtype=torch.bfloat16)]",
        ),
        wrap(
            r"""
@tilelang.jit(out_idx=[-1], target="cuda")
def make_kernel(N):
    @T.prim_func
    def binary(A:T.Tensor((257,513),"bfloat16"),C:T.Tensor((257,513),"bfloat16"),B:T.Tensor((257,513),"bfloat16")):
        with T.Kernel(T.ceildiv(N,256),threads=256) as bx:
            for tx in T.Parallel(256):
                i=bx*256+tx
                if i<N:
                    r=i//513; c=i%513; B[r,c]=A[r,c]*T.bfloat16(0.75)+C[r,c]*T.bfloat16(1.5)
    return binary
""",
            "257*513",
        ),
        "bf16",
        "tilelang",
    ),
    (
        "silu_fp16",
        ref(
            "    def forward(self,x): return torch.nn.functional.silu(x)",
            "    return [torch.randn(131071,dtype=torch.float16)]",
        ),
        wrap(
            r"""
@tilelang.jit(out_idx=[-1], target="cuda")
def make_kernel(N):
    @T.prim_func
    def silu(A:T.Tensor((N,),"float16"),B:T.Tensor((N,),"float16")):
        with T.Kernel(T.ceildiv(N,256),threads=256) as bx:
            for tx in T.Parallel(256):
                i=bx*256+tx
                if i<N:
                    v=T.float32(A[i]); B[i]=v/(T.float32(1)+T.exp(-v))
    return silu
""",
            "131071",
        ),
        "fp16",
        "tilelang",
    ),
    (
        "rope_fp32",
        ref(
            "    def forward(self,x,c,s):\n        h=x.shape[-1]//2; return torch.cat((x[:,:h]*c-x[:,h:]*s,x[:,h:]*c+x[:,:h]*s),-1)",
            "    \n    x=torch.randn(257,128); a=torch.randn(257,64); return [x,torch.cos(a),torch.sin(a)]",
        ),
        wrap(
            r"""
@tilelang.jit(out_idx=[-1], target="cuda")
def make_kernel(R,H):
    @T.prim_func
    def rope(A:T.Tensor((R,H*2),"float32"),C:T.Tensor((R,H),"float32"),S:T.Tensor((R,H),"float32"),B:T.Tensor((R,H*2),"float32")):
        with T.Kernel(T.ceildiv(R*H,256),threads=256) as bx:
            for tx in T.Parallel(256):
                i=bx*256+tx
                if i<R*H:
                    r=i//H; c=i%H; a=A[r,c]; b=A[r,c+H]; B[r,c]=a*C[r,c]-b*S[r,c]; B[r,c+H]=b*C[r,c]+a*S[r,c]
    return rope
""",
            "257,64",
        ),
        "fp32",
        "tilelang",
    ),
    (
        "kv_gather_fp16",
        ref(
            "    def forward(self,cache,slots): return cache[slots]",
            "    return [torch.randn(4096,256,dtype=torch.float16),torch.randint(0,4096,(257,),dtype=torch.int64)]",
        ),
        wrap(
            r"""
@tilelang.jit(out_idx=[-1], target="cuda")
def make_kernel(TOK,D):
    @T.prim_func
    def gather(A:T.Tensor((4096,D),"float16"),S:T.Tensor((TOK,),"int64"),B:T.Tensor((TOK,D),"float16")):
        with T.Kernel(TOK,threads=256) as bx:
            for tx in T.Parallel(256):
                if tx<D: B[bx,tx]=A[S[bx],tx]
    return gather
""",
            "257,256",
        ),
        "fp16",
        "tilelang",
    ),
    (
        "attention_merge_fp32",
        ref(
            "    def forward(self,v1,s1,v2,s2):\n        m=torch.maximum(s1,s2); a=torch.exp(s1-m); b=torch.exp(s2-m); return (v1*a[:,None]+v2*b[:,None])/(a+b)[:,None]",
            "    return [torch.randn(257,128),torch.randn(257),torch.randn(257,128),torch.randn(257)]",
        ),
        wrap(
            r"""
@tilelang.jit(out_idx=[-1], target="cuda")
def make_kernel(R,D):
    @T.prim_func
    def merge(V1:T.Tensor((R,D),"float32"),S1:T.Tensor((R,),"float32"),V2:T.Tensor((R,D),"float32"),S2:T.Tensor((R,),"float32"),B:T.Tensor((R,D),"float32")):
        with T.Kernel(R,threads=128) as bx:
            for tx in T.Parallel(128):
                hi=T.max(S1[bx],S2[bx]); a=T.exp(S1[bx]-hi); b=T.exp(S2[bx]-hi); B[bx,tx]=(V1[bx,tx]*a+V2[bx,tx]*b)/(a+b)
    return merge
""",
            "257,128",
        ),
        "fp32",
        "tilelang",
    ),
    (
        "matmul_fp16",
        ref(
            "    def forward(self,a,b): return a@b",
            "    return [torch.randn(32,64,dtype=torch.float16),torch.randn(64,48,dtype=torch.float16)]",
        ),
        wrap(
            r"""
@tilelang.jit(out_idx=[-1], target="cuda")
def make_kernel(M,N,K):
    @T.prim_func
    def matmul(A:T.Tensor((M,K),"float16"),W:T.Tensor((K,N),"float16"),B:T.Tensor((M,N),"float16")):
        with T.Kernel(M,threads=64) as bx:
            for tx in T.Parallel(64):
                if tx<N:
                    acc=T.alloc_local((1,),"float32"); acc[0]=T.float32(0)
                    for k in T.serial(K): acc[0]+=T.float32(A[bx,k])*T.float32(W[k,tx])
                    B[bx,tx]=acc[0]
    return matmul
""",
            "32,48,64",
        ),
        "fp16",
        "tilelang",
    ),
    (
        "noncontig_fp32",
        ref("    def forward(self,x): return x*2-0.5", "    return [torch.randn(513,257).t()]"),
        wrap(
            r"""
@tilelang.jit(out_idx=[-1], target="cuda")
def make_kernel(R,C):
    @T.prim_func
    def transform(A:T.StridedTensor((R,C),(1,R),"float32"),B:T.Tensor((R,C),"float32")):
        with T.Kernel(T.ceildiv(R*C,256),threads=256) as bx:
            for tx in T.Parallel(256):
                i=bx*256+tx
                if i<R*C: B[i//C,i%C]=A[i//C,i%C]*T.float32(2)-T.float32(0.5)
    return transform
""",
            "257,513",
        ),
        "fp32",
        "tilelang",
    ),
    (
        "rmsnorm_fp32",
        ref(
            "    def forward(self,x,w): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*w",
            "    return [torch.randn(32,1024),torch.randn(1024)]",
        ),
        wrap(
            r"""
@tilelang.jit(out_idx=[-1], target="cuda")
def make_kernel(R,H):
    @T.prim_func
    def rms(A:T.Tensor((R,H),"float32"),W:T.Tensor((H,),"float32"),B:T.Tensor((R,H),"float32")):
        with T.Kernel(R,threads=1) as bx:
            acc=T.alloc_local((1,),"float32"); acc[0]=T.float32(0)
            for k in T.serial(H): acc[0]+=A[bx,k]*A[bx,k]
            inv=T.rsqrt(acc[0]/T.float32(H)+T.float32(1e-6))
            for k in T.serial(H): B[bx,k]=A[bx,k]*inv*W[k]
    return rms
""",
            "32,1024",
        ),
        "fp32",
        "tilelang",
    ),
    (
        "torch_only_rejected",
        ref("    def forward(self,x): return x+1", "    return [torch.randn(1024)]"),
        "import torch\nfrom torch import nn\nclass ModelNew(nn.Module):\n    def forward(self,x): return x+1\n",
        "fp32",
        "tilelang",
    ),
]


def run(case, split=False):
    name, reference, kernel, precision, backend = case
    payload = {
        "task_id": f"wide-tilelang-{name}-{uuid.uuid4().hex[:8]}",
        "reference_code": reference,
        "kernel_code": kernel,
        "backend": backend,
        "precision": precision,
        "num_correct_trials": 3,
        "num_perf_trials": 10,
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
    start = time.time()
    req = urllib.request.Request(URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=360) as response:
            result = json.load(response)
    except Exception as exc:
        return {"name": name, "transport_error": repr(exc), "wall_s": round(time.time() - start, 2)}
    md = result.get("metadata") or {}
    prof = md.get("profiling") or {}
    return {
        "name": name,
        "backend": backend,
        "status": result.get("status"),
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "decoy": result.get("decoy_kernel"),
        "runtime_ms": result.get("kernel_runtime"),
        "speedup": result.get("speedup"),
        "profile_kernels": prof.get("kernel_count"),
        "precompiled": md.get("precompiled_artifact_used"),
        "error": result.get("error_message"),
        "wall_s": round(time.time() - start, 2),
    }


if __name__ == "__main__":
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, c) for c in CASES]
        for f in concurrent.futures.as_completed(futures):
            item = f.result()
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    print("SPLIT=" + json.dumps(run(CASES[0], split=True), ensure_ascii=False))
    print("RESULTS_JSON=" + json.dumps(sorted(results, key=lambda x: x["name"]), ensure_ascii=False))
