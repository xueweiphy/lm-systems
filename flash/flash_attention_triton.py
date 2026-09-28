# FlashAttention-2 forward pass as a Triton kernel, with causal masking (CS336 A2 §4.2.2 b, c)
# Needs a CUDA GPU and triton.
import torch
import triton
import triton.language as tl

from flash.flash_attention import flash_bwd_c


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal:tl.constexpr,
):

    query_tile_index = tl.program_id ( 0)
    batch_index = tl.program_id ( 1)

    Q_block_ptr = tl.make_block_ptr (
        Q_ptr + batch_index * stride_qb,
        shape =( N_QUERIES, D),
        strides = ( stride_qq , stride_qd ),
        offsets = ( query_tile_index * Q_TILE_SIZE , 0 ),
        block_shape =( Q_TILE_SIZE, D),
        order = ( 1, 0 ),
    )

    K_block_ptr = tl.make_block_ptr (
        K_ptr + batch_index * stride_kb,
        shape =( N_KEYS, D),
        strides = ( stride_kk , stride_kd ),
        offsets = ( 0 , 0 ),
        block_shape =( K_TILE_SIZE, D),
        order = ( 1, 0 ),
    )

    V_block_ptr = tl.make_block_ptr (
        V_ptr + batch_index * stride_vb,
        shape =( N_KEYS, D),
        strides = ( stride_vk , stride_vd ),
        offsets = ( 0 , 0 ),
        block_shape =( K_TILE_SIZE, D),
        order = ( 1, 0 ),
    )

    O_block_ptr = tl.make_block_ptr (
        O_ptr + batch_index * stride_ob,
        shape =( N_QUERIES, D),
        strides = ( stride_oq , stride_od ),
        offsets = ( query_tile_index * Q_TILE_SIZE , 0 ),
        block_shape =( Q_TILE_SIZE, D),
        order = ( 1, 0 ),
    )

    L_block_ptr = tl.make_block_ptr (
        L_ptr + batch_index * stride_lb,
        shape =( N_QUERIES,),
        strides = ( stride_lq , ),
        offsets = ( query_tile_index * Q_TILE_SIZE,  ),
        block_shape =( Q_TILE_SIZE,),
        order = ( 0, ),
    )

    # initilize  a buffer
    Oi = tl.zeros ( ( Q_TILE_SIZE, D), dtype = tl.float32)

    mi = tl.full ( ( Q_TILE_SIZE, ), float('-inf'), dtype = tl.float32 )
    li = tl.zeros ( ( Q_TILE_SIZE, ), dtype = tl.float32 )

    Qi = tl.load ( Q_block_ptr, boundary_check = ( 0, 1), padding_option = 'zero')

    n_tiles = tl.cdiv ( N_KEYS, K_TILE_SIZE )
    if is_causal :
        n_tiles = tl.cdiv ( ( query_tile_index + 1 ) * Q_TILE_SIZE, K_TILE_SIZE )

    for jj in range ( n_tiles ) :

        Ki = tl.load ( K_block_ptr, boundary_check= (0,1), padding_option = 'zero')
        Vi = tl.load ( V_block_ptr, boundary_check= (0,1), padding_option = 'zero')

        Si = tl.dot ( Qi, tl.trans ( Ki ) ) * scale

        if is_causal :
            q_pos = query_tile_index * Q_TILE_SIZE + tl.arange ( 0, Q_TILE_SIZE )     # (Bq,)
            k_pos = jj * K_TILE_SIZE + tl.arange ( 0, K_TILE_SIZE )

            visible =q_pos[:, None] >= k_pos[None, :]
            Si = tl.where ( visible, Si, Si - 1.0e6 )

        Simax = tl.max ( Si, axis = 1 )
        mnew = tl.maximum ( mi  , Simax)
        mexp = tl.exp(mi - mnew)
        pi = tl.exp( Si - mnew[:, None])

        li = mexp * li + tl.sum ( pi, axis = 1)

        Oi = tl.dot ( pi.to( Vi.dtype), Vi , acc = mexp[:, None]* Oi, )

        mi = mnew

        K_block_ptr = K_block_ptr.advance ( ( K_TILE_SIZE, 0 ))
        V_block_ptr = V_block_ptr.advance ( ( K_TILE_SIZE, 0 ))

    Oi = Oi/ li [:, None]
    Li = mi + tl.log( li )

    tl.store ( O_block_ptr, Oi.to ( O_ptr.type.element_ty ), boundary_check = ( 0,1))
    tl.store ( L_block_ptr , Li, boundary_check = (0,) )

class FlashForwardTriton (torch.autograd.Function):

    @staticmethod
    def forward ( ctx, Q, K, V,  is_causal = False ) :
        # hard coded
        Bq = 16
        Bk = 16

        ctx.is_causal = is_causal

        Tk, dk = K.shape[-2:]
        Bsize, Tq = Q.shape[:-1]

        scale = dk**-0.5
        O = torch.empty_like ( Q )
        L = torch.empty ( Q.shape[:-1], device = Q.device, dtype = torch.float32 )

        flash_fwd_kernel[( triton.cdiv ( Tq,Bq ) , Bsize )]( Q, K, V ,O, L,
                        Q.stride(0), Q.stride(1), Q.stride(2),
                        K.stride(0), K.stride(1), K.stride(2),
                        V.stride(0), V.stride(1), V.stride(2),
                        O.stride(0), O.stride(1), O.stride(2),
                        L.stride(0), L.stride (1),
                        Tq, Tk,
                        scale,
                        D=  dk,
                        Q_TILE_SIZE= Bq,
                        K_TILE_SIZE= Bk,
                        is_causal = is_causal,)

        ctx.save_for_backward ( L, Q, K,  V, O)

        return O

    @staticmethod
    def backward ( ctx , dO ):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_bwd_c ( Q, K, V, O, dO, L, ctx.is_causal )
        return dQ, dK, dV, None
