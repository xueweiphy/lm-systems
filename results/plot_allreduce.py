import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list ( csv.DictReader ( open ( "allreduce.csv" ) ) )
d = {}
for r in rows :
    d.setdefault ( int ( r["world_size"] ), [] ).append (
        ( int ( r["bytes"] ), float ( r["seconds"] ) ) )

COL   = { 2 : "#2a78d6", 4 : "#eb6834", 6 : "#1baf7a" }
INK   = "#0b0b0b"
MUTED = "#52514e"

fig, ax = plt.subplots ( 1, 2, figsize = ( 11, 4.3 ) )

for a in ax :
    a.set_facecolor ( "#fcfcfb" )
    a.grid ( True, which = "both", lw = 0.5, color = "#dcdcd8" )
    a.set_axisbelow ( True )
    for s in ( "top", "right" ) : a.spines[s].set_visible ( False )
    for s in ( "left", "bottom" ) : a.spines[s].set_color ( "#c9c9c4" )
    a.tick_params ( colors = MUTED, labelsize = 9 )
fig.patch.set_facecolor ( "#fcfcfb" )

for N in sorted ( d ) :
    x = [ b / 1e6      for b, t in d[N] ]
    y = [ t * 1e3      for b, t in d[N] ]
    ax[0].plot ( x, y, "-o", color = COL[N], lw = 2, ms = 6, label = f"{N} processes" )
    ax[0].annotate ( f"N={N}", ( x[-1], y[-1] ), xytext = ( 6, 0 ),
                     textcoords = "offset points", color = COL[N],
                     fontsize = 9, va = "center" )

ax[0].set_xscale ( "log" ) ; ax[0].set_yscale ( "log" )
ax[0].set_xlabel ( "message size  (MB)", color = MUTED, fontsize = 10 )
ax[0].set_ylabel ( "all-reduce time  (ms)", color = MUTED, fontsize = 10 )
ax[0].set_title ( "Time grows linearly with size above ~10 MB",
                  color = INK, fontsize = 11, loc = "left", pad = 10 )
ax[0].legend ( frameon = False, fontsize = 9, labelcolor = MUTED, loc = "upper left" )

for N in sorted ( d ) :
    x  = [ b / 1e6 for b, t in d[N] ]
    bw = [ 2 * ( N - 1 ) * b / t / 1e9 for b, t in d[N] ]
    ax[1].plot ( x, bw, "-o", color = COL[N], lw = 2, ms = 6, label = f"{N} processes" )
    dy = { 2 : -9, 4 : 9, 6 : 0 }[N]
    ax[1].annotate ( f"N={N}", ( x[-1], bw[-1] ), xytext = ( 6, dy ),
                     textcoords = "offset points", color = COL[N],
                     fontsize = 9, va = "center" )

ax[1].axhline ( 14.5, ls = "--", lw = 1.2, color = MUTED )
ax[1].annotate ( "~14.5 GB/s shared-memory ceiling", ( 1.3, 15.2 ),
                 color = MUTED, fontsize = 9 )
ax[1].set_xscale ( "log" )
ax[1].set_ylim ( 0, 18 )
ax[1].set_xlabel ( "message size  (MB)", color = MUTED, fontsize = 10 )
ax[1].set_ylabel ( "aggregate bandwidth  2(N-1)D/t  (GB/s)", color = MUTED, fontsize = 10 )
ax[1].set_title ( "All world sizes hit the same ceiling",
                  color = INK, fontsize = 11, loc = "left", pad = 10 )
ax[1].legend ( frameon = False, fontsize = 9, labelcolor = MUTED, loc = "lower right" )

fig.suptitle ( "gloo all-reduce on one Mac mini (10 cores, 32 GB) - CS336 A2 section 5.1",
               color = INK, fontsize = 12, x = 0.012, ha = "left", y = 0.99 )
fig.tight_layout ( rect = [ 0, 0, 1, 0.94 ] )
fig.savefig ( "allreduce_scaling.png", dpi = 160, facecolor = fig.get_facecolor() )
print ( "saved" )
