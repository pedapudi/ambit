"""Live 3-D point cloud — the real projected reservoir (ctx.xyz) baked into a
drag-to-rotate / scroll-to-zoom canvas. Vanilla JS, no deps; reads CSS tokens so
it re-skins on theme swap. Ports the full study live-view feature set: depth-sorted
points coloured by genre cluster, a #-points slider to thin the field, and the
rotating x/y/z origin gnomon at lower-left. An OPTIONAL kNN-edge overlay (off by
default, toggled by a checkbox) wires the visible points by their k-nearest-neighbor
graph — same-cluster edges in the cluster colour at low alpha, cross-cluster
"bridge" edges neutral. Returns a `script` field (collected by build_report)."""

from __future__ import annotations

import numpy as np

from ..render import figure

_JS = r"""
var cv=document.getElementById("ambit-live-canvas");
if(cv&&cv.dataset.init!=="1"){var ctx=cv.getContext&&cv.getContext("2d");
if(ctx){cv.dataset.init="1";
var DEG=Math.PI/180,yaw=35*DEG,pitch=22*DEG,zoom=1,ZMIN=0.5,ZMAX=5;
var rm=(window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches);
var auto=!rm,SPIN=7*DEG,lastIdle=0,RESUME=2600,tok={};
var PAL_TOK=["--accent","--good","--bad","--caution","--ink-soft"],pal=[];
var rng=document.getElementById("d3live-range"),cnt=document.getElementById("d3live-count");
var edgeBox=document.getElementById("d3live-edges");
var VIS=rng?(+rng.value):PTS.length;
function readTok(){var cs=getComputedStyle(document.documentElement);var fb=(getComputedStyle(cv).color||"").trim();
 tok.faint=cs.getPropertyValue("--ink-faint").trim()||fb;tok.accent=cs.getPropertyValue("--accent").trim()||tok.faint;
 tok.bridge=cs.getPropertyValue("--ink").trim()||tok.faint;
 pal=[];for(var k=0;k<NCL;k++){var c=cs.getPropertyValue(PAL_TOK[k%PAL_TOK.length]).trim();pal.push(c||tok.accent);}}
var cssW=600,cssH=420,dpr=1,AS=0.66;
function resize(){var w=cv.clientWidth||(cv.parentNode&&cv.parentNode.clientWidth)||600;cssW=Math.max(240,w);
 cssH=Math.round(cssW*AS);dpr=window.devicePixelRatio||1;cv.style.height=cssH+"px";
 cv.width=Math.round(cssW*dpr);cv.height=Math.round(cssH*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
function gnomon(cx,cy,cyw,syw,cp,sp){
 var g=Math.min(cssW,cssH)*0.12,ox=g+12,oy=cssH-g-12,ax=[[1,0,0,"x"],[0,1,0,"y"],[0,0,1,"z"]],a;
 ctx.strokeStyle=tok.faint;ctx.fillStyle=tok.faint;ctx.globalAlpha=0.62;ctx.lineWidth=1;
 ctx.font="9px ui-monospace,Menlo,Consolas,monospace";ctx.textAlign="center";ctx.textBaseline="middle";
 for(a=0;a<3;a++){var v=ax[a],x1=v[0]*cyw+v[2]*syw,z1=-v[0]*syw+v[2]*cyw,y1=v[1]*cp-z1*sp;
  ctx.beginPath();ctx.moveTo(ox,oy);ctx.lineTo(ox+x1*g,oy-y1*g);ctx.stroke();
  ctx.fillText(v[3],ox+x1*g*1.2,oy-y1*g*1.2);}
 ctx.beginPath();ctx.arc(ox,oy,1.6,0,6.2831853);ctx.fill();
 ctx.globalAlpha=1;ctx.textAlign="start";ctx.textBaseline="alphabetic";}
function edges(scr,n){
 // overlay only when toggled on; wire the visible points by the baked kNN graph
 if(!(edgeBox&&edgeBox.checked)||!EDG.length)return;
 // group edges by colour: one path per cluster (same-cluster, low alpha) + one bridge path
 var nc=NCL,sameP=[],e,a,b;for(var i=0;i<nc;i++)sameP.push([]);var brP=[];
 for(e=0;e<EDG.length;e++){var ea=EDG[e][0],eb=EDG[e][1];if(ea>=n||eb>=n)continue;
  var ca=PTS[ea][3],cb=PTS[eb][3];if(ca===cb)sameP[ca%nc].push([scr[ea],scr[eb]]);else brP.push([scr[ea],scr[eb]]);}
 for(i=0;i<nc;i++){var seg=sameP[i];if(!seg.length)continue;
  ctx.strokeStyle=pal[i]||tok.accent;ctx.globalAlpha=0.13;ctx.lineWidth=0.6;ctx.beginPath();
  for(e=0;e<seg.length;e++){a=seg[e][0];b=seg[e][1];ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);}ctx.stroke();}
 if(brP.length){ctx.strokeStyle=tok.bridge;ctx.globalAlpha=0.30;ctx.lineWidth=0.7;ctx.beginPath();
  for(e=0;e<brP.length;e++){a=brP[e][0];b=brP[e][1];ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);}ctx.stroke();}
 ctx.globalAlpha=1;}
function draw(){ctx.clearRect(0,0,cssW,cssH);
 var cx=cssW/2,cy=cssH/2,sc=Math.min(cssW,cssH)*0.40*zoom,zf=Math.pow(zoom,0.35);
 var cyw=Math.cos(yaw),syw=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),pr=[],i;
 var n=Math.max(1,Math.min(PTS.length,VIS|0));
 // screen-space projection per source index (for edge endpoints), unsorted
 var scr=new Array(n);
 for(i=0;i<n;i++){var p=PTS[i];var x1=p[0]*cyw+p[2]*syw,z1=-p[0]*syw+p[2]*cyw,y1=p[1]*cp-z1*sp,z2=p[1]*sp+z1*cp;
  var sx=cx+x1*sc,sy=cy-y1*sc;scr[i]=[sx,sy,z2];pr.push([sx,sy,z2,p[3]]);}
 edges(scr,n);
 pr.sort(function(a,b){return a[2]-b[2];});
 var zmin=Infinity,zmax=-Infinity;for(i=0;i<pr.length;i++){if(pr[i][2]<zmin)zmin=pr[i][2];if(pr[i][2]>zmax)zmax=pr[i][2];}
 var zr=(zmax-zmin)||1;
 for(i=0;i<pr.length;i++){var q=pr[i],nd=(q[2]-zmin)/zr,r,al,col;
  col=pal[q[3]]||tok.accent;          /* cluster colour */
  r=(0.8+nd*1.4)*zf;                   /* near larger, far smaller */
  al=0.22+nd*0.55;                     /* near brighter, far dimmer */
  ctx.globalAlpha=al;ctx.fillStyle=col;ctx.beginPath();ctx.arc(q[0],q[1],r,0,6.2831853);ctx.fill();}
 ctx.globalAlpha=1;gnomon(cx,cy,cyw,syw,cp,sp);}
var drag=false,lx=0,ly=0,pin=false,pd=0,pz=1;
function pt(e){var r=cv.getBoundingClientRect(),t=(e.touches&&e.touches[0])||e;return [t.clientX-r.left,t.clientY-r.top];}
function td(e){var a=e.touches[0],b=e.touches[1];return Math.hypot(a.clientX-b.clientX,a.clientY-b.clientY);}
function cz(z){return Math.max(ZMIN,Math.min(ZMAX,z));}
function down(e){if(e.touches&&e.touches.length===2){pin=true;drag=false;pd=td(e)||1;pz=zoom;auto=false;
  if(e.cancelable)e.preventDefault();return;}drag=true;auto=false;cv.style.cursor="grabbing";
  var p=pt(e);lx=p[0];ly=p[1];if(e.cancelable)e.preventDefault();}
function move(e){if(pin&&e.touches&&e.touches.length===2){zoom=cz(pz*(td(e)/pd));draw();if(e.cancelable)e.preventDefault();return;}
  if(!drag)return;var p=pt(e);yaw+=(p[0]-lx)*0.01;pitch+=(p[1]-ly)*0.01;var L=85*DEG;
  if(pitch>L)pitch=L;if(pitch<-L)pitch=-L;lx=p[0];ly=p[1];draw();if(e.cancelable)e.preventDefault();}
function up(){pin=false;if(!drag)return;drag=false;cv.style.cursor="grab";lastIdle=performance.now();}
function whl(e){e.preventDefault();auto=false;lastIdle=performance.now();zoom=cz(zoom*Math.exp(-e.deltaY*0.0014));draw();}
cv.addEventListener("pointerdown",down);window.addEventListener("pointermove",move);window.addEventListener("pointerup",up);
cv.addEventListener("touchstart",down,{passive:false});window.addEventListener("touchmove",move,{passive:false});
window.addEventListener("touchend",up);cv.addEventListener("wheel",whl,{passive:false});
var prev=0;function frame(t){var dt=(t-prev)/1000;prev=t;
 if(auto&&!drag&&!pin){yaw+=SPIN*dt;draw();}
 else if(!rm&&!drag&&!pin&&!auto&&lastIdle&&(t-lastIdle)>RESUME){auto=true;}requestAnimationFrame(frame);}
function showCount(){if(cnt)cnt.textContent=(VIS|0).toLocaleString()+" of "+PTS.length.toLocaleString()+" points";}
if(rng){rng.addEventListener("input",function(){VIS=+rng.value;showCount();draw();});}
if(edgeBox){edgeBox.addEventListener("change",function(){draw();});}
new MutationObserver(function(){readTok();draw();}).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
if(window.ResizeObserver){new ResizeObserver(function(){resize();}).observe(cv.parentNode||cv);}
window.addEventListener("resize",resize);
readTok();showCount();resize();requestAnimationFrame(function(t){prev=t;frame(t);});
}}
"""


_PAL_TOK = ["--accent", "--good", "--bad", "--caution", "--ink-soft"]


@figure
def fig_d3_live(ctx):
    P = np.asarray(ctx.xyz, dtype=float)
    X = np.asarray(ctx.es.X, dtype=np.float32)   # L2-normalized reservoir
    m = len(P)
    labels = np.asarray(ctx.labels)
    if m > 8000:  # the live cloud never needs more than a few thousand points
        idx = np.linspace(0, m - 1, 8000).astype(int)
        P = P[idx]
        X = X[idx]
        labels = labels[idx]
    c = P - P.mean(0)
    s = float(np.abs(c).max()) or 1.0
    C = c / s

    # map the distinct genres to indices 0..NCL-1 (stable, sorted)
    genres = sorted({str(g) for g in labels.tolist()})
    gidx = {g: i for i, g in enumerate(genres)}
    cluster = np.array([gidx[str(g)] for g in labels], dtype=int)
    NCL = len(genres)

    # shuffle ONCE so the slider thins an unbiased, genre-mixed prefix
    rng = np.random.default_rng(0)
    order = rng.permutation(len(C))
    C = C[order]
    X = X[order]
    cluster = cluster[order]

    # kNN edge list over a capped prefix of the SHUFFLED points (edges over all
    # 8000 would be too many). The overlay draws an edge only when the toggle is
    # on and both endpoints fall under the slider's visible count, so building it
    # over the first ~1500 shuffled points keeps it dense yet cheap.
    ncap = min(1500, len(C))
    U = X[:ncap]
    kk = min(6, max(1, ncap - 1))
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    nbr = np.argpartition(-S, kth=kk, axis=1)[:, :kk]
    eset = set()
    for i in range(ncap):
        for j in nbr[i]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b:
                eset.add((a, b))
    edges = sorted(eset)

    pts = "[" + ",".join("[%.3f,%.3f,%.3f,%d]" % (C[i, 0], C[i, 1], C[i, 2], int(cluster[i]))
                         for i in range(len(C))) + "]"
    eb = "[" + ",".join("[%d,%d]" % (a, b) for a, b in edges) + "]"

    total = len(C)
    vis = max(1, min(total, 6000))   # default ~6000 visible
    script = ("(function(){var PTS=%s;var EDG=%s;var NCL=%d;%s})();" % (pts, eb, NCL, _JS))

    ctrl = ('<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:6px;'
            'font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--ink-faint)">'
            '<span style="display:inline-flex;align-items:center;gap:10px">'
            f'samples <input id="d3live-range" type="range" min="1" max="{total}" value="{vis}" '
            'style="flex:0 0 220px;accent-color:var(--accent)"> '
            '<span id="d3live-count"></span></span>'
            '<label style="display:inline-flex;align-items:center;gap:5px;cursor:pointer;user-select:none">'
            '<input id="d3live-edges" type="checkbox" style="accent-color:var(--accent)"> kNN edges</label>'
            '</div>')
    canvas = (
        ctrl +
        '<canvas id="ambit-live-canvas" role="img" '
        'aria-label="Interactive 3-D point cloud of the projected reservoir coloured by genre cluster; drag to rotate, scroll or pinch to zoom; an optional kNN-edge overlay toggled by a checkbox wires the visible points; an x/y/z origin gnomon at lower-left shows orientation" '
        'style="display:block;width:100%;height:auto;touch-action:none;cursor:grab;background:transparent">'
        'Your browser does not support the canvas element; the static 3-D triptych shows the same cloud.</canvas>'
        '<div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10px;letter-spacing:.06em;'
        'color:var(--ink-faint);margin-top:6px;text-align:center">drag to rotate · scroll / pinch to zoom · auto-spins when idle · toggle kNN edges to wire the cloud</div>')

    legend = "".join(
        '<span><i style="background:var(%s)"></i> %s</span>'
        % (_PAL_TOK[i % len(_PAL_TOK)], genres[i]) for i in range(NCL)
    ) + (
        '<span><i style="background:var(--ink)"></i> bridge edge (cross-cluster)</span>'
        '<span><i class="dash"></i> kNN edges (toggle) · x/y/z origin gnomon</span>'
    )

    return {
        "num": "3D · live", "order": 19, "name": "Live 3-D cloud (drag · zoom · kNN edges)", "tech": "canvas · drag/zoom · clusters · knn",
        "why": "The projected reservoir as a turnable solid — the same xyz the triptych shows, live, with each point coloured by its genre cluster; flip on the kNN-edge overlay to wire the visible points by their k-nearest-neighbor graph (same-cluster edges in the cluster colour, neutral bridges across clusters). The lower-left x/y/z gnomon tracks orientation as you rotate.",
        "svg": canvas, "script": script,
        "legend": legend,
        "reveal": "<b>Reveals:</b> how the genre clusters sit in the occupied volume — rotate and zoom from any angle, depth grades each cluster's colour (near brighter, far dimmer), and the samples slider thins the field. Toggle <b>kNN edges</b> to overlay the connective tissue — where each genre coheres and where neutral bridges stitch clusters together; the gnomon keeps you oriented as the cloud turns.",
        "cls": "fig-mid",
    }
