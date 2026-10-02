# flash_benchmarking (A2 4.2.2): fwd / bwd / e2e for flash, naive, compiled, sdpa
# run from the repo root:  python -m flash.bench_flash   (writes flash_benchmark.csv)
import torch
import torch.nn.functional as F
import pandas as pd
from triton.testing import do_bench
from flash.flash_attention_triton import FlashForwardTriton

torch._functorch.config.donated_buffer = False   # needed for retain_graph with compiled bwd
torch._dynamo.config.recompile_limit = 128        # one compile per shape


def softmax ( xin , dim = -1, ) :
    shifted  = xin - torch.max ( xin, dim = dim , keepdim=True ).values
    se = shifted.exp()
    expsum = torch.sum ( se, dim = dim , keepdim= True )
    return se/ expsum


def Attention ( Q, K , V ) :
    T, dk = K.shape[-2:]
    masked = torch.triu ( torch.ones ( ( T, T ) , dtype = torch.bool, device=Q.device), diagonal = 1 )
    qk = Q@ torch.transpose ( K, -2,-1 )
    qk = qk.masked_fill ( masked , float('-inf'))
    return softmax ( qk * dk**-0.5 )  @ V


def bench(fn, Q, K, V, dO):
    fwd = do_bench ( lambda:  fn ( Q, K, V ))
    Ores = fn ( Q, K, V )
    bwd = do_bench ( lambda: Ores.backward ( dO, retain_graph = True ), grad_to_none= [Q, K, V] )
    e2e = do_bench ( lambda: fn ( Q, K, V ).backward ( dO ) ,grad_to_none= [Q, K, V]  )
    return fwd, bwd, e2e


def flash_tile ( b ) :                # flash with tile size Bq = Bk = b
    def fn ( Q, K, V ) :
        FlashForwardTriton.Bq = FlashForwardTriton.Bk = b
        return FlashForwardTriton.apply ( Q, K, V, True )
    return fn

tiles = [16, 32, 64, 128]
sdpa = lambda Q, K, V: F.scaled_dot_product_attention(Q[:, None], K[:, None], V[:, None], is_causal=True)[:, 0]
attnC = torch.compile(Attention)
impls = { f"flash{b}": flash_tile ( b ) for b in tiles }
impls.update ( {"naive": Attention, "compiled": attnC, "sdpa": sdpa} )

Ts = [2**i for i in range(7, 17)]
ds = [16, 32, 64, 128]
dtypes = [torch.bfloat16, torch.float32]
rows = []
gpu = torch.cuda.get_device_name()
print ( gpu )

for dt in dtypes:
    for d in ds:
        for T in Ts:
            Q, K, V, dO = ( torch.randn ( 1, T, d, device = 'cuda', dtype = dt, requires_grad = True ) for _ in range ( 4 ) )
            for name, fn in impls.items() :
                try:
                    fwd, bwd, e2e =  bench(fn, Q, K, V, dO)
                except Exception as e:      # OOM, or Triton OutOfResources for big tiles
                    print(name, str(e)[:80])
                    fwd = bwd = e2e = float('nan')
                rows.append((name, str(dt), T, d, fwd, bwd, e2e, gpu))
                print (name, str(dt), T, d, fwd, bwd, e2e)
            del Q, K, V, dO
            torch.cuda.empty_cache()
        # save after each d, so a crash keeps what is done
        pd.DataFrame(rows, columns=["impl", "dtype", "T", "d", "fwd", "bwd", "e2e", "gpu"]).to_csv("flash_benchmark.csv", index=False)
