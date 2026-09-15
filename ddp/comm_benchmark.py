import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import statistics


def setup ( rank, world_size ) :
    os.environ ["MASTER_ADDR"] ="localhost"
    os.environ ["MASTER_PORT"] = "29500"
    dist.init_process_group ( "gloo", rank = rank, world_size = world_size )

def worker ( rank , world_size , sizes ) :
    setup ( rank, world_size )

    warm_up = 5

    for n in sizes :
        data = torch.randn ( (n), dtype = torch.float32)

        for _ in range ( warm_up ) :
            dist.all_reduce ( data )




        times = []
        n_trial = 10
        

        for _ in range ( n_trial ) :
            dist.barrier()

            t0 = time.time()
            dist.all_reduce ( data, async_op = False )
            times.append (  time.time() - t0 ) 

        #print (f"rank {rank} data (after all-reduce ): {data}")
        #print (f"rank {rank} time {dt}")

        dt = statistics.median(times)

        t= torch.tensor( [dt ])
        
        dist.all_reduce ( t, op = dist.ReduceOp.MAX)
        if rank == 0:
            print ( f"world {world_size}  n {n} max {t.item():.6f} s")
            with open ( "allreduce.csv", "a" ) as f :
                f.write ( f"{world_size},{n*4},{t.item():.6f}\n" )
        

    dist.destroy_process_group()
    

if __name__ =="__main__":

    with open ( "allreduce.csv", "w" ) as f :
        f.write ( "world_size,bytes,seconds\n" )

    sizes = [ 1024*1024//4, 10*1024*1024//4, 100*1024*1024//4, 1024*1024*1024//4 ]
    for world_size in ( 2, 4, 6 ) : 
        mp.spawn ( fn = worker, args = ( world_size, sizes) , nprocs = world_size, join = True)





















