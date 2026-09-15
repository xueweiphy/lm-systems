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



class DDP ( torch.nn.Module ) :
    def __init__ ( self, module ) :
        super().__init__()
        self.module = module 

        for p in module.parameters() :
            dist.broadcast ( p.data, src = 0 )         

    def forward (self,*args, **kwargs ) :
        return self.module ( *args, **kwargs  )  

    def finish_gradient_synchronization ( self ) :

        for p in self.module.parameters():
            if p.grad is None :
                continue

            dist.all_reduce ( p.grad, async_op = False )
            p.grad /= dist.get_world_size() 



def worker ( rank , world_size  ) :

    setup ( rank, world_size )
    torch.manual_seed ( 0)  

    model = torch.nn.Sequential ( torch.nn.Linear ( 10, 16) , torch.nn.ReLU(), 
                                    torch.nn.Linear ( 16, 5) ) 


    ddp = DDP ( model ) 

    opt = torch.optim.SGD( ddp.parameters(), lr=0.1 )

    batch_total = 32
    trainNum = 5 

    x = torch.randn (batch_total, 10 ) 
    y = torch.randn ( batch_total, 5 ) 


    b = batch_total// world_size 

    xs  = x [ rank * b : ( rank +1 )*b ]
    ys  = y [ rank * b : ( rank +1 )*b ]


    if True : 
        print ( rank, list ( ddp.parameters() ) [0] [0] [:3] )

    for step in range ( trainNum) :

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss ( ddp ( xs), ys )    

        loss.backward ()
        
        ddp.finish_gradient_synchronization ()

        opt.step()







    l = loss.detach().clone()
    dist.all_reduce ( l ) 
    if rank == 0 :
        print  ( step, ( l / world_size ).item() ) 

    print ( "final", list ( ddp.parameters() ) [0] [0] [:3] )


    dist.destroy_process_group()
    

if __name__ =="__main__":

    world_size = 4

    mp.spawn ( fn = worker, args = ( world_size, ) , nprocs = world_size, join = True)





















