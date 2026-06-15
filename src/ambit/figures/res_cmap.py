"""RES 07 — Local crowding cloud. The projected reservoir (ctx.xyz) as an
interactive, drag-to-rotate / scroll-to-zoom canvas, recolored by each item's local
crowding score (the signed, multiscale robust z of its neighborhood concentration vs
the dataset's own field). The color runs open -> elevated -> crowded along
green -> amber -> red, keyed to crowding rank so adjacent levels stay distinct; and
the SHAPE encodes it too — open/typical items are flat dots, crowded items rise into
little pyramids (taller and brighter the more crowded), so the crowded regions read
even at a glance and from any angle.

Spatial companion to RES 06: RES 06 says how much and what kind of crowding (global
vs pocketed, vs the isotropic reference); RES 07 says where it sits. Both read the
same `local_anisotropy.for_ctx` result. Vanilla JS, no deps; reads CSS tokens so it
re-skins on theme swap (returns a `script` field collected by build_report)."""

from __future__ import annotations

import numpy as np

from ..render import figure
from .. import local_anisotropy

_JS = r"""
var cv=document.getElementById("ambit-crowd-canvas");
if(cv&&cv.dataset.init!=="1"){var ctx=cv.getContext&&cv.getContext("2d");
if(ctx){cv.dataset.init="1";
var DEG=Math.PI/180,yaw=35*DEG,pitch=22*DEG,zoom=1,ZMIN=0.5,ZMAX=5;
var rm=(window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches);
var auto=!rm,SPIN=7*DEG,lastIdle=0,RESUME=2600,tok={};
var rng=document.getElementById("res07-range"),cnt=document.getElementById("res07-count");
var edgeBox=document.getElementById("res07-edges");
var VIS=rng?(+rng.value):PTS.length;
function hx(s){s=(s||"").trim();if(s.charAt(0)==="#"){if(s.length===4)s="#"+s[1]+s[1]+s[2]+s[2]+s[3]+s[3];
 return [parseInt(s.substr(1,2),16),parseInt(s.substr(3,2),16),parseInt(s.substr(5,2),16)];}
 var m=s.match(/[\d.]+/g);return m?[+m[0],+m[1],+m[2]]:[136,136,136];}
function lerp(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}
function rs(c){return "rgb("+(c[0]|0)+","+(c[1]|0)+","+(c[2]|0)+")";}
function readTok(){var cs=getComputedStyle(document.documentElement);var fb=(getComputedStyle(cv).color||"").trim();
 function rd(n){return hx(cs.getPropertyValue(n).trim()||fb);}
 tok.good=rd("--good");tok.caution=rd("--caution");tok.bad=rd("--bad");tok.faint=rd("--ink-faint");
 tok.faintStr=cs.getPropertyValue("--ink-faint").trim()||fb;}
function colorFor(q){if(q>=0.5)return lerp(tok.caution,tok.bad,(q-0.5)/0.5);
 return lerp(tok.faint,tok.good,0.3+0.6*((0.5-q)/0.5));}
var cssW=600,cssH=420,dpr=1,AS=0.66;
function resize(){var w=cv.clientWidth||(cv.parentNode&&cv.parentNode.clientWidth)||600;cssW=Math.max(240,w);
 cssH=Math.round(cssW*AS);dpr=window.devicePixelRatio||1;cv.style.height=cssH+"px";
 cv.width=Math.round(cssW*dpr);cv.height=Math.round(cssH*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
function gnomon(cx,cy,cyw,syw,cp,sp){
 var g=Math.min(cssW,cssH)*0.12,ox=g+12,oy=cssH-g-12,ax=[[1,0,0,"x"],[0,1,0,"y"],[0,0,1,"z"]],a;
 ctx.strokeStyle=tok.faintStr;ctx.fillStyle=tok.faintStr;ctx.globalAlpha=0.62;ctx.lineWidth=1;
 ctx.font="9px ui-monospace,Menlo,Consolas,monospace";ctx.textAlign="center";ctx.textBaseline="middle";
 for(a=0;a<3;a++){var v=ax[a],x1=v[0]*cyw+v[2]*syw,z1=-v[0]*syw+v[2]*cyw,y1=v[1]*cp-z1*sp;
  ctx.beginPath();ctx.moveTo(ox,oy);ctx.lineTo(ox+x1*g,oy-y1*g);ctx.stroke();
  ctx.fillText(v[3],ox+x1*g*1.2,oy-y1*g*1.2);}
 ctx.beginPath();ctx.arc(ox,oy,1.6,0,6.2831853);ctx.fill();
 ctx.globalAlpha=1;ctx.textAlign="start";ctx.textBaseline="alphabetic";}
function pyramid(x,y,r,col,pf){
 var dk=[col[0]*0.62,col[1]*0.62,col[2]*0.62];
 var ax=x,ay=y-r*1.3,blx=x-r,bly=y+r*0.72,brx=x+r,bry=y+r*0.72,bmx=x,bmy=y+r*1.0;
 ctx.fillStyle=rs(col);ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(blx,bly);ctx.lineTo(bmx,bmy);ctx.closePath();ctx.fill();
 ctx.fillStyle=rs(dk);ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(brx,bry);ctx.lineTo(bmx,bmy);ctx.closePath();ctx.fill();
 if(pf){ctx.strokeStyle=rs(tok.bad);ctx.globalAlpha=Math.min(1,ctx.globalAlpha+0.25);ctx.lineWidth=0.9;
  ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(blx,bly);ctx.lineTo(brx,bry);ctx.closePath();ctx.stroke();}}
function edges(scr,n){
 // overlay only when toggled on; wire the visible points by their native kNN graph.
 // an edge between two crowded points (both q>=0.5) is tinted bad, the rest neutral.
 if(!(edgeBox&&edgeBox.checked)||!EDG.length)return;
 var hot=[],neu=[],e,a,b;
 for(e=0;e<EDG.length;e++){var ea=EDG[e][0],eb=EDG[e][1];if(ea>=n||eb>=n)continue;
  if(PTS[ea][3]>=0.5&&PTS[eb][3]>=0.5)hot.push(e);else neu.push(e);}
 ctx.lineWidth=0.6;
 if(neu.length){ctx.strokeStyle=rs(tok.faint);ctx.globalAlpha=0.11;ctx.beginPath();
  for(e=0;e<neu.length;e++){a=scr[EDG[neu[e]][0]];b=scr[EDG[neu[e]][1]];ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);}ctx.stroke();}
 if(hot.length){ctx.strokeStyle=rs(tok.bad);ctx.globalAlpha=0.22;ctx.beginPath();
  for(e=0;e<hot.length;e++){a=scr[EDG[hot[e]][0]];b=scr[EDG[hot[e]][1]];ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);}ctx.stroke();}
 ctx.globalAlpha=1;}
function draw(){ctx.clearRect(0,0,cssW,cssH);
 var cx=cssW/2,cy=cssH/2,sc=Math.min(cssW,cssH)*0.40*zoom,zf=Math.pow(zoom,0.35);
 var cyw=Math.cos(yaw),syw=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),pr=[],i;
 var n=Math.max(1,Math.min(PTS.length,VIS|0)),scr=new Array(n);
 for(i=0;i<n;i++){var p=PTS[i];var x1=p[0]*cyw+p[2]*syw,z1=-p[0]*syw+p[2]*cyw,y1=p[1]*cp-z1*sp,z2=p[1]*sp+z1*cp;
  var sx=cx+x1*sc,sy=cy-y1*sc;scr[i]=[sx,sy];pr.push([sx,sy,z2,p[3],p[4]]);}
 edges(scr,n);
 pr.sort(function(a,b){return a[2]-b[2];});
 var zmin=Infinity,zmax=-Infinity;for(i=0;i<pr.length;i++){if(pr[i][2]<zmin)zmin=pr[i][2];if(pr[i][2]>zmax)zmax=pr[i][2];}
 var zr=(zmax-zmin)||1;
 for(i=0;i<pr.length;i++){var o=pr[i],nd=(o[2]-zmin)/zr,q=o[3],pf=o[4],col=colorFor(q);
  if(q>=0.5){ctx.globalAlpha=0.5+nd*0.45;pyramid(o[0],o[1],(0.6+1.05*q+nd*0.45)*zf,col,pf);}
  else{ctx.globalAlpha=0.26+nd*0.46;ctx.fillStyle=rs(col);ctx.beginPath();
   ctx.arc(o[0],o[1],(0.7+nd*1.2)*zf,0,6.2831853);ctx.fill();}}
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


@figure
def fig_res_cmap(ctx):
    P = np.asarray(ctx.xyz, dtype=float)
    X = np.asarray(ctx.es.X, dtype=np.float32)        # reservoir vectors, for the kNN edges
    m = len(P)

    la = local_anisotropy.for_ctx(ctx)
    s = np.nan_to_num(np.asarray(la.score, float), nan=0.0)
    # crowding rank in [0,1] (0 most open … 1 most crowded) — even color spread
    q = np.empty(m, float)
    q[np.argsort(s, kind="stable")] = np.arange(m) / max(m - 1, 1)
    pf = np.zeros(m, np.int8)
    for p in la.pockets:
        pf[np.asarray(p.members, int)] = 1
    gc = float(la.global_crowding)
    crowded_share = float((q >= 0.5).mean())

    # the live cloud never needs more than a few thousand points
    if m > 8000:
        idx = np.linspace(0, m - 1, 8000).astype(int)
        P, X, q, pf = P[idx], X[idx], q[idx], pf[idx]
    c = P - P.mean(0)
    sc = float(np.abs(c).max()) or 1.0
    C = c / sc

    # shuffle once so the slider thins an unbiased prefix
    rng = np.random.default_rng(0)
    order = rng.permutation(len(C))
    C, X, q, pf = C[order], X[order], q[order], pf[order]

    # kNN edge overlay (toggleable): wire a capped prefix of the shuffled points by
    # their native cosine kNN. Built over the first ~1500 shuffled points; the overlay
    # draws an edge only when both endpoints fall under the slider's visible count.
    ncap = min(1500, len(C))
    U = X[:ncap] / np.maximum(np.linalg.norm(X[:ncap], axis=1, keepdims=True), 1e-12)
    kk = min(6, max(1, ncap - 1))
    Sm = U @ U.T
    np.fill_diagonal(Sm, -np.inf)
    nbr = np.argpartition(-Sm, kth=kk, axis=1)[:, :kk]
    eset = set()
    for i in range(ncap):
        for j in nbr[i]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b:
                eset.add((a, b))
    eb = "[" + ",".join("[%d,%d]" % (a, b) for a, b in sorted(eset)) + "]"

    pts = "[" + ",".join(
        "[%.3f,%.3f,%.3f,%.3f,%d]" % (C[i, 0], C[i, 1], C[i, 2], q[i], int(pf[i]))
        for i in range(len(C))) + "]"
    total = len(C)
    vis = max(1, min(total, 5000))
    script = "(function(){var PTS=%s;var EDG=%s;%s})();" % (pts, eb, _JS)

    ctrl = ('<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:6px;'
            'font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--ink-faint)">'
            '<span style="display:inline-flex;align-items:center;gap:10px">'
            f'samples <input id="res07-range" type="range" min="1" max="{total}" value="{vis}" '
            'style="flex:0 0 220px;accent-color:var(--accent)"> <span id="res07-count"></span></span>'
            '<label style="display:inline-flex;align-items:center;gap:5px;cursor:pointer;user-select:none">'
            '<input id="res07-edges" type="checkbox" style="accent-color:var(--accent)"> kNN edges</label></div>')
    canvas = (
        ctrl +
        '<canvas id="ambit-crowd-canvas" role="img" '
        'aria-label="Interactive 3-D cloud of the projected reservoir recolored by local crowding; open items '
        'are flat green dots, crowded items rise into amber-to-red pyramids (taller and brighter the more '
        'crowded); drag to rotate, scroll or pinch to zoom, auto-spins when idle; an optional kNN-edge overlay '
        'toggled by a checkbox wires the visible points by their nearest-neighbor graph; an x/y/z origin gnomon '
        'at lower-left shows orientation" '
        'style="display:block;width:100%;height:auto;touch-action:none;cursor:grab;background:transparent">'
        'Your browser does not support the canvas element.</canvas>'
        '<div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10px;letter-spacing:.06em;'
        'color:var(--ink-faint);margin-top:6px;text-align:center;line-height:1.6">'
        'drag to rotate · scroll / pinch to zoom · auto-spins when idle · toggle kNN edges to wire the cloud<br>'
        'flat dot = open · taller pyramid = more crowded<br>'
        '<span style="color:var(--caution)">position is the PCA projection (global variance) — read crowding '
        'from color and shape, not from how clumped the dots look</span></div>')

    legend = ('<span><i class="g"></i> open (flat dot)</span>'
              '<span><i style="background:var(--caution);clip-path:polygon(50% 0,100% 100%,0 100%)"></i> '
              'elevated (pyramid)</span>'
              '<span><i class="r" style="clip-path:polygon(50% 0,100% 100%,0 100%)"></i> '
              'crowded (taller, brighter pyramid)</span>'
              + ('<span><i class="r" style="border:1px solid var(--paper)"></i> ringed = crowded pocket</span>'
                 if la.pockets else '')
              + '<span><i style="height:2px;background:var(--ink-faint);border:none;align-self:center"></i> '
                'kNN edges (toggle)</span>'
              + '<span><i class="dash"></i> x/y/z origin gnomon</span>')

    read = (f"{len(la.pockets)} crowded pocket(s)" if la.pockets else
            ("globally crowded — the whole cloud rides high relative to the isotropic reference"
             if gc > 8.0 else "roomy — little local crowding structure"))

    return {
        "num": "RES 07", "order": 95.5,
        "name": "Local crowding cloud", "tech": "canvas · drag/zoom · crowding shape · kNN edges",
        "why": "The projected reservoir as a turnable solid, recolored and reshaped by each item's local "
               "crowding score (signed multiscale z of its neighborhood concentration vs the dataset's own "
               "field). Open items stay flat dots; crowded items rise into pyramids — taller, brighter, redder "
               "the more crowded. Read crowding from the color and shape: the position is a PCA projection of "
               "the global variance and cannot show 768-d local crowding (only a few percent of each item's "
               "true neighbors survive the projection). Toggle the kNN-edge overlay to wire each point to its "
               "nearest neighbors — edges between two crowded points are tinted, the rest neutral.",
        "svg": canvas, "script": script,
        "legend": legend,
        "reveal": (f"<b>Reveals:</b> whether crowding is spatially coherent or an even wash ({read}). A field of "
                   "tall red pyramids concentrated in one region is a localized pocket; an even spread is global "
                   "crowding. The green dots are the <em>least-crowded items relative to this dataset</em> — not "
                   "necessarily roomy in absolute terms (RES 06 shows the absolute level). Position is a PCA "
                   "projection and does not encode crowding — read it from color and shape. The kNN edges link "
                   "true 768-d neighbors, so they fan across the projection rather than staying local — that "
                   "scatter is the projection limit made visible. Drag to rotate, scroll to zoom, thin with the "
                   "slider, toggle kNN edges."),
        "cls": "fig-mid",
    }
