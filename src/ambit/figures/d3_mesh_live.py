"""3D 04 — Live kNN manifold mesh. A subsample of the projected reservoir wired by
its k-nearest-neighbor graph, as a drag-to-rotate / scroll-to-zoom canvas (vanilla
JS, theme-adaptive). Nodes are coloured by genre cluster; same-cluster edges take
that cluster's colour at low alpha, cross-cluster "bridge" edges read neutral and
stand out. A #-nodes slider thins the mesh coherently; an x/y/z gnomon tracks
orientation."""

from __future__ import annotations

import numpy as np

from ..render import figure

_JS = r"""
var cv=document.getElementById("ambit-mesh-canvas");
if(cv&&cv.dataset.init!=="1"){var ctx=cv.getContext&&cv.getContext("2d");
if(ctx){cv.dataset.init="1";
var DEG=Math.PI/180,yaw=35*DEG,pitch=22*DEG,zoom=1,ZMIN=0.5,ZMAX=5;
var rm=(window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches);
var auto=!rm,SPIN=6*DEG,lastIdle=0,RESUME=2600,tok={},proj=[];
var rng=document.getElementById("mesh-range"),cnt=document.getElementById("mesh-count");
var NMAX=PTS.length,NVIS=NMAX;
var PAL=["--accent","--good","--bad","--caution","--ink-soft"];
function readTok(){var cs=getComputedStyle(document.documentElement);var fb=(getComputedStyle(cv).color||"").trim();
 tok.faint=cs.getPropertyValue("--ink-faint").trim()||fb;
 tok.bridge=cs.getPropertyValue("--ink").trim()||tok.faint;
 tok.pal=[];for(var k=0;k<PAL.length;k++){var c=cs.getPropertyValue(PAL[k]).trim();tok.pal.push(c||tok.faint);}}
function colOf(ci){return tok.pal[ci%tok.pal.length];}
var cssW=600,cssH=400,dpr=1,AS=0.66;
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
function draw(){ctx.clearRect(0,0,cssW,cssH);
 var cx=cssW/2,cy=cssH/2,sc=Math.min(cssW,cssH)*0.40*zoom,zf=Math.pow(zoom,0.35);
 var cyw=Math.cos(yaw),syw=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),i,n=NVIS;
 proj.length=0;
 for(i=0;i<n;i++){var p=PTS[i];var x1=p[0]*cyw+p[2]*syw,z1=-p[0]*syw+p[2]*cyw,y1=p[1]*cp-z1*sp,z2=p[1]*sp+z1*cp;
  proj.push([cx+x1*sc,cy-y1*sc,z2,p[3]]);}
 // group edges by colour: one path per cluster (same-cluster, low alpha) + one bridge path
 var nc=tok.pal.length,sameP=[],e,a,b;for(i=0;i<nc;i++)sameP.push([]);var brP=[];
 for(e=0;e<EDG.length;e++){var ea=EDG[e][0],eb=EDG[e][1];if(ea>=n||eb>=n)continue;
  var ca=PTS[ea][3],cb=PTS[eb][3];if(ca===cb)sameP[ca%nc].push([proj[ea],proj[eb]]);else brP.push([proj[ea],proj[eb]]);}
 for(i=0;i<nc;i++){var seg=sameP[i];if(!seg.length)continue;
  ctx.strokeStyle=colOf(i);ctx.globalAlpha=0.13;ctx.lineWidth=0.6;ctx.beginPath();
  for(e=0;e<seg.length;e++){a=seg[e][0];b=seg[e][1];ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);}ctx.stroke();}
 if(brP.length){ctx.strokeStyle=tok.bridge;ctx.globalAlpha=0.30;ctx.lineWidth=0.7;ctx.beginPath();
  for(e=0;e<brP.length;e++){a=brP[e][0];b=brP[e][1];ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);}ctx.stroke();}
 ctx.globalAlpha=1;
 var zmin=Infinity,zmax=-Infinity;for(i=0;i<n;i++){if(proj[i][2]<zmin)zmin=proj[i][2];if(proj[i][2]>zmax)zmax=proj[i][2];}
 var zr=(zmax-zmin)||1;
 for(i=0;i<n;i++){var q=proj[i],nd=(q[2]-zmin)/zr,r=(0.8+nd*1.2)*zf,al=0.45+nd*0.5;
  ctx.globalAlpha=al;ctx.fillStyle=colOf(q[3]);ctx.beginPath();ctx.arc(q[0],q[1],r,0,6.2831853);ctx.fill();}
 ctx.globalAlpha=1;gnomon(cx,cy,cyw,syw,cp,sp);}
function applyN(){var v=rng?(+rng.value):NMAX;NVIS=Math.max(2,Math.min(NMAX,v));
 if(cnt)cnt.textContent=NVIS.toLocaleString()+" of "+NMAX.toLocaleString()+" nodes";draw();}
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
if(rng){rng.addEventListener("input",applyN);}
var prev=0;function frame(t){var dt=(t-prev)/1000;prev=t;
 if(auto&&!drag&&!pin){yaw+=SPIN*dt;draw();}
 else if(!rm&&!drag&&!pin&&!auto&&lastIdle&&(t-lastIdle)>RESUME){auto=true;}requestAnimationFrame(frame);}
new MutationObserver(function(){readTok();draw();}).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
if(window.ResizeObserver){new ResizeObserver(function(){resize();}).observe(cv.parentNode||cv);}
window.addEventListener("resize",resize);
readTok();applyN();resize();requestAnimationFrame(function(t){prev=t;frame(t);});
}}
"""


@figure
def fig_d3_mesh_live(ctx):
    P3 = np.asarray(ctx.xyz, dtype=float)
    X = np.asarray(ctx.es.X, dtype=np.float32)   # L2-normalized reservoir
    labels = np.asarray(getattr(ctx, "labels", None))
    m = len(P3)
    n = min(1500, m)                              # cap nodes so the edge mesh stays legible/fast
    idx = np.linspace(0, m - 1, n).astype(int) if m > n else np.arange(m)
    P = P3[idx]
    U = X[idx]
    lab = labels[idx] if labels is not None and labels.shape == (m,) else np.array([""] * n)
    c = P - P.mean(0)
    s = float(np.abs(c).max()) or 1.0
    C = c / s

    # genre -> cluster index 0..C-1 (stable order over the subsample's labels)
    genres = sorted({str(v) for v in lab})
    gmap = {g: i for i, g in enumerate(genres)}
    cl = np.array([gmap.get(str(v), 0) for v in lab], dtype=int)

    kk = min(6, max(1, n - 1))
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    nbr = np.argpartition(-S, kth=kk, axis=1)[:, :kk]
    eset = set()
    for i in range(n):
        for j in nbr[i]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b:
                eset.add((a, b))
    edges = sorted(eset)

    # shuffle nodes once so the slider thins evenly across the cloud; remap edge indices
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    C = C[order]
    cl = cl[order]
    edges = sorted(((int(inv[a]), int(inv[b])) if inv[a] < inv[b] else (int(inv[b]), int(inv[a])))
                   for a, b in edges)

    pts = "[" + ",".join("[%.3f,%.3f,%.3f,%d]" % (C[i, 0], C[i, 1], C[i, 2], int(cl[i]))
                         for i in range(n)) + "]"
    eb = "[" + ",".join("[%d,%d]" % (a, b) for a, b in edges) + "]"
    script = "(function(){var PTS=" + pts + ";var EDG=" + eb + ";" + _JS + "})();"

    # default: thin to a legible mesh when the subsample is large
    nvis = min(n, 700) if n > 700 else n
    ctrl = ('<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;'
            'font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--ink-faint)">'
            f'#nodes <input id="mesh-range" type="range" min="2" max="{n}" value="{nvis}" '
            'style="flex:0 0 220px;accent-color:var(--accent)"> '
            '<span id="mesh-count"></span></div>')

    canvas = (
        ctrl +
        '<canvas id="ambit-mesh-canvas" role="img" '
        'aria-label="Interactive 3-D kNN manifold mesh of a reservoir subsample, nodes coloured by genre cluster; drag to rotate, scroll or pinch to zoom, slider thins the mesh" '
        'style="display:block;width:100%;height:auto;touch-action:none;cursor:grab;background:transparent">'
        'Your browser does not support the canvas element; the static 3-D views show the same structure.</canvas>'
        '<div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10px;letter-spacing:.06em;'
        f'color:var(--ink-faint);margin-top:6px;text-align:center">drag to rotate · scroll / pinch to zoom · '
        f'{len(edges):,} edges over {n:,} nodes · {len(genres)} clusters</div>')

    pal = ["--accent", "--good", "--bad", "--caution", "--ink-soft"]
    leg_clusters = "".join(
        f'<span><i style="background:var({pal[i % len(pal)]})"></i> {g}</span>'
        for i, g in enumerate(genres))
    legend = (leg_clusters +
              '<span><i style="background:var(--ink)"></i> bridge edge (cross-cluster)</span>'
              '<span><i class="dash"></i> x/y/z origin gnomon</span>')

    return {
        "num": "3D 04 · live", "order": 23, "name": "Live kNN manifold mesh (drag · zoom)", "tech": "canvas · knn mesh · clusters",
        "why": "A subsample of the projected reservoir wired by its k-nearest-neighbor graph and coloured by genre cluster — rotate to read connectivity, watch same-cluster edges fill in their colour while cross-cluster bridges stand out in neutral. The slider thins the mesh.",
        "svg": canvas, "script": script,
        "legend": legend,
        "reveal": "<b>Reveals:</b> the connective tissue of the cloud, turnable in 3-D and coloured by cluster — where each genre coheres, where neutral <b>bridge</b> edges stitch clusters together, and where points hang off alone. Drag the slider to thin the mesh and watch which structure survives.",
        "cls": "fig-mid",
    }
