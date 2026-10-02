# FlashAttention-2 in pure PyTorch (CS336 A2 §4.2.2): tiled forward (a) and the
# recomputation backward (flash_backward), shared with the Triton version.
import torch


def flash_bwd ( Q, K, V, O, dO, L, is_causal ) :
    dtype = Q.dtype
    Q, K, V, O, dO = ( t.float () for t in ( Q, K, V, O, dO ) )
    T, dk = K.shape [-2:]
    S = Q@ torch.transpose ( K, -2,-1 ) * dk**-0.5
    if is_causal :
        Tq = Q.shape[-2]
        keep = torch.arange ( Tq, device = S.device )[:, None] >= torch.arange ( T, device = S.device )[None, :]
        S = torch.where ( keep, S, S - 1.0e6 )

    P = ( S - L [...,None] ).exp()
    dV = torch.transpose ( P, -2,-1) @ dO

    dP = dO@ torch.transpose ( V, -2,-1)
    D = torch.sum ( O * dO , dim = -1, keepdim = True )

    dS  = P * ( dP - D )
    dQ = dS @ K  * dk**-0.5
    dK = torch.transpose ( dS, -2, -1) @ Q * dk**-0.5

    return dQ.to ( dtype ), dK.to ( dtype ), dV.to ( dtype )
# on a Mac without a working torch.compile, use:  flash_bwd_c = flash_bwd
flash_bwd_c = torch.compile ( flash_bwd )


class FlashForward (torch.autograd.Function):

    @staticmethod
    def forward ( ctx, Q, K, V,  is_causal = False ) :
        # hard coded
        Bq = 16
        Bk = 16

        Tk, dk = K.shape[-2:]
        Tq = Q.shape[-2]

        O = []
        L = []
        for qii in range ( 0, Tq, Bq) :
            Qi = Q[:, qii : qii + Bq ]
            mi = torch.full ( (*Q.shape[:-2], Bq, 1), -torch.inf , device = Q.device )
            li = torch.zeros ( (*Q.shape[:-2], Bq, 1) , device = Q.device )
            Oi = torch.zeros ( (*Q.shape[:-2], Bq, dk) , device = Q.device )

            for kii in range ( 0, Tk, Bk) :
                Ki =  K[ :, kii: kii+ Bk ]
                Vi =  V[ :, kii: kii+ Bk ]
                qk = Qi @ torch.transpose ( Ki, -2,-1 ) * dk**-0.5
                mnew =  torch.maximum ( mi, qk.amax ( dim = -1 , keepdim = True) )
                pi = (qk - mnew).exp()
                mexp = (mi - mnew).exp()
                li =  mexp* li + torch.sum ( pi, dim = -1 , keepdim= True)
                Oi = mexp * Oi +  pi @ Vi

                mi = mnew

            Oi = Oi / li
            Li = mi + li.log()
            O.append ( Oi)
            L.append ( Li )


        O = torch.cat ( O, dim = -2 )
        L = torch.cat ( L, dim = -2 )[..., 0]
        ctx.is_causal = is_causal
        ctx.save_for_backward ( L, Q, K,  V, O)

        return O


    @staticmethod
    def backward ( ctx , dO ):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_bwd_c ( Q, K, V, O, dO, L, ctx.is_causal )
        return dQ, dK, dV, None
