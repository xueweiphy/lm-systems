"""Ring all-reduce, simulated on one machine (CS336 A2 territory; Gibiansky 2017).

Problem.  N nodes each hold a vector x_i of length D.  After all-reduce every
node holds  x_0 + x_1 + ... + x_{N-1}.  The naive way (every node sends its
whole vector to every other node) moves (N-1)·D per node.  The ring moves
2·(N-1)/N·D per node — about 2·D, independent of N — and that is why it is
what NCCL and Horovod use.

Algorithm.  Split every vector into N chunks.  Nodes sit in a ring: node i
sends only to node (i+1) mod N and receives only from node (i-1) mod N.

  Phase 1, scatter-reduce, N-1 steps.  At step k, node i
      sends     its chunk  (i - k)     mod N   to node i+1
      receives  the chunk  (i - k - 1) mod N   from node i-1, and ADDS it to its own.
  Afterwards node i holds the fully summed chunk (i + 1) mod N — one chunk each.

  Phase 2, all-gather, N-1 steps.  At step k, node i
      sends     its chunk  (i + 1 - k) mod N   (already complete)   to node i+1
      receives  the chunk  (i - k)     mod N   from node i-1, and OVERWRITES its own.
  Afterwards every node holds every summed chunk.

Total 2·(N-1) steps, each moving one chunk of D/N per node.

Simulation.  "Sending" is reading from the neighbour's list entry.  Two
things learned writing it:

  * np.array_split returns VIEWS.  Splitting the caller's arrays and then
    doing += on the chunks writes straight through into the caller's data —
    the first version silently scrambled `vec`, and the naive reference
    computed from the same `vec` afterwards summed the scrambled rows, so
    "expected 5" printed 35.  Hence x.copy() before splitting.  Rule: anything
    that mutates in place gets its own copy of the input unless in-place is
    the documented contract.

  * The in-step ordering is safe as written.  At step k node i receives into
    chunk (i-k-1) and sends chunk (i-k) — never the same chunk — so the
    sequential loop over nodes never forwards a value received earlier in the
    same step.  That is what the (i-k) stagger buys; a real implementation,
    sending and receiving concurrently, relies on the same fact.

Handles D not divisible by N (np.array_split).

    ring_allreduce ( [ x_0, ..., x_{N-1} ] )  ->  [ s, s, ..., s ]   with  s = Σ x_i
"""
import numpy as np


def naive_allreduce ( nodes ) :
    s = sum ( nodes )
    return [ s.copy() for _ in nodes ]

def ScatterReduce ( chunks ) :
    Nn = len ( chunks ) 
    for kk in range ( Nn - 1 )  :
        for ii in range ( Nn ) :
            chunks[ ( ii+1 ) % Nn ] [ ( ii - kk) % Nn] += chunks[ ii  ] [ ( ii - kk) % Nn]
    return chunks

def AllGather ( chunks ) :
    Nn = len ( chunks ) 
    for kk in range ( Nn - 1 )  :
        for ii in range ( Nn ) :
            chunks[ii ] [ ( ii - kk) % Nn] = chunks[  ( ii-1 ) % Nn ] [ ( ii - kk) % Nn]
    return chunks
    
def ring_allreduce ( nodes ) :
    """nodes: list of N equal-length 1-D arrays.  Returns a list of N arrays, all equal to the sum."""
    N = len ( nodes )
    chunks = [ np.array_split ( x.copy(), N ) for x in nodes ]     # chunks[i][j] = chunk j on node i
    ScatterReduce ( chunks )      # N-1 steps, receive and ADD
    AllGather ( chunks )          # N-1 steps, receive and OVERWRITE
    return [ np.concatenate ( c ) for c in chunks ]
