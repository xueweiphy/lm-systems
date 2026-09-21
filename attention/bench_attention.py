# Benchmark naive PyTorch attention: forward/backward time and memory saved for backward (CS336 A2 §4.1.1)
# run:  python attention/bench_attention.py   -> prints one row per config, writes pytorch_attention.csv
import torch
import timeit

device = 'cuda' if torch.cuda.is_available () else 'cpu'


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
    out = softmax ( qk * dk**-0.5 )  @ V
    return out


def sync() :
    if torch.cuda.is_available ()  :
        torch.cuda.synchronize ()


def bench ( T, D, B = 8 , warm_up = 5, nrun = 100 ) :
    Q0= torch.randn ( ( B, T, D), device  = device, requires_grad = True)
    K0= torch.randn ( ( B, T, D), device  = device, requires_grad = True)
    V0= torch.randn ( ( B, T, D), device  = device, requires_grad = True)

    att = None
    y = None

    for _ in range ( warm_up ) :
        att = None
        att =  Attention ( Q0, K0 , V0 ) 
    sync()

    dts = []
    for _ in range ( nrun ) :
        att = None
        sync()
        t0 = timeit.default_timer ()
        att =  Attention ( Q0, K0 , V0 )
        sync()
        dts.append (  timeit.default_timer () - t0 )
    fwd = torch.tensor ( dts ) * 1e3

    m1 = torch.cuda.memory_allocated()          # memory in use before backward starts
    torch.cuda.reset_peak_memory_stats()

    dts = []
    for _ in range ( nrun ) :
        att = None ; y = None
        att =  Attention ( Q0, K0 , V0 )
        y = att.sum()
        sync()
        t0 = timeit.default_timer ()
        y.backward()
        sync()
        dts.append (  timeit.default_timer () - t0 )
    bwd = torch.tensor ( dts ) * 1e3
    m1max = torch.cuda.max_memory_allocated()

    return fwd.mean().item(), fwd.std().item(), bwd.mean().item(), bwd.std().item(), m1/1024**2, m1max/1024**2


if __name__ == '__main__' :
    print ( device, torch.cuda.get_device_name ( 0 ) if device == 'cuda' else '' )
    if device == 'cuda' :
        print ( 'free, total MiB:', [ m // 1024**2 for m in torch.cuda.mem_get_info () ] )

    B = 8
    Dlist = [16, 32, 64, 128]
    Tlist = [256, 1024, 4096, 8192, 16384]

    f = open ( 'pytorch_attention.csv', 'w' )
    f.write ( 'T,D,fwd_ms,fwd_std,bwd_ms,bwd_std,mem_before_bwd_MiB,peak_MiB\n' )
    for T in Tlist :
        for D in Dlist :
            try :
                r = bench ( T, D, B )
                row = f"{T},{D}," + ",".join ( f"{x:.3f}" for x in r )
            except RuntimeError :               # OutOfMemoryError is a subclass; also catches the NVML assert on MIG
                row = f"{T},{D},OOM,,,,,"
            torch.cuda.empty_cache ()
            print ( row, f"  leftover {torch.cuda.memory_allocated()/1024**2:.1f} MiB" , flush = True )
            f.write ( row + '\n' ) ; f.flush ()
    f.close ()
