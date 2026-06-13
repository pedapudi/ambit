"""Live 3-D point cloud — the real projected reservoir (ctx.xyz) baked into a
drag-to-rotate / scroll-to-zoom canvas. Vanilla JS, no deps; reads CSS tokens so
it re-skins on theme swap. Ports the full study live-view feature set: depth-sorted
restrained palette, an outlier-tail fade, and the rotating x/y/z origin gnomon at
lower-left. Returns a `script` field (collected by build_report)."""

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
function readTok(){var cs=getComputedStyle(document.documentElement);var fb=(getComputedStyle(cv).color||"").trim();
 tok.faint=cs.getPropertyValue("--ink-faint").trim()||fb;tok.accent=cs.getPropertyValue("--accent").trim()||tok.faint;}
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
function draw(){ctx.clearRect(0,0,cssW,cssH);
 var cx=cssW/2,cy=cssH/2,sc=Math.min(cssW,cssH)*0.40*zoom,zf=Math.pow(zoom,0.35);
 var cyw=Math.cos(yaw),syw=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),pr=[],i,n=PTS.length;
 for(i=0;i<n;i++){var p=PTS[i];var x1=p[0]*cyw+p[2]*syw,z1=-p[0]*syw+p[2]*cyw,y1=p[1]*cp-z1*sp,z2=p[1]*sp+z1*cp;
  pr.push([cx+x1*sc,cy-y1*sc,z2,p[3]]);}
 pr.sort(function(a,b){return a[2]-b[2];});
 var zmin=Infinity,zmax=-Infinity;for(i=0;i<pr.length;i++){if(pr[i][2]<zmin)zmin=pr[i][2];if(pr[i][2]>zmax)zmax=pr[i][2];}
 var zr=(zmax-zmin)||1;
 for(i=0;i<pr.length;i++){var q=pr[i],nd=(q[2]-zmin)/zr,r,al,col;
  if(q[3]===1){col=tok.accent;r=1.4+nd*1.2;al=0.55+nd*0.4;}
  else if(q[3]===2){col=tok.faint;r=0.6+nd*0.7;al=0.10+nd*0.18;}
  else{col=tok.faint;r=0.8+nd*1.0;al=0.14+nd*0.34;}
  r*=zf;ctx.globalAlpha=al;ctx.fillStyle=col;ctx.beginPath();ctx.arc(q[0],q[1],r,0,6.2831853);ctx.fill();}
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
new MutationObserver(function(){readTok();draw();}).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
if(window.ResizeObserver){new ResizeObserver(function(){resize();}).observe(cv.parentNode||cv);}
window.addEventListener("resize",resize);
readTok();resize();requestAnimationFrame(function(t){prev=t;frame(t);});
}}
"""


@figure
def fig_d3_live(ctx):
    P = np.asarray(ctx.xyz, dtype=float)
    m = len(P)
    if m > 5000:  # the live cloud never needs more than a few thousand points
        idx = np.linspace(0, m - 1, 5000).astype(int)
        P = P[idx]
        kd = None if ctx.knn_dist is None else ctx.knn_dist[idx]
    else:
        kd = ctx.knn_dist
    c = P - P.mean(0)
    s = float(np.abs(c).max()) or 1.0
    C = c / s
    if kd is not None:
        dens = -kd.mean(1)                       # smaller mean kNN distance = denser
        acc = dens >= np.quantile(dens, 0.90)    # the dense core carries the accent
    else:
        acc = np.zeros(len(C), dtype=bool)
    rad = np.linalg.norm(C, axis=1)
    outlier = (rad >= np.quantile(rad, 0.98)) & ~acc   # the sparse tail, drawn extra-faint
    kind = np.where(acc, 1, np.where(outlier, 2, 0)).astype(int)
    pts = "[" + ",".join("[%.3f,%.3f,%.3f,%d]" % (C[i, 0], C[i, 1], C[i, 2], int(kind[i]))
                         for i in range(len(C))) + "]"
    script = "(function(){var PTS=" + pts + ";" + _JS + "})();"
    canvas = (
        '<canvas id="ambit-live-canvas" role="img" '
        'aria-label="Interactive 3-D point cloud of the projected reservoir; drag to rotate, scroll or pinch to zoom; an x/y/z origin gnomon at lower-left shows orientation" '
        'style="display:block;width:100%;height:auto;touch-action:none;cursor:grab;background:transparent">'
        'Your browser does not support the canvas element; the static 3-D triptych shows the same cloud.</canvas>'
        '<div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10px;letter-spacing:.06em;'
        'color:var(--ink-faint);margin-top:6px;text-align:center">drag to rotate · scroll / pinch to zoom · auto-spins when idle</div>')
    return {
        "num": "3D · live", "order": 19, "name": "Live 3-D cloud (drag · zoom)", "tech": "canvas · drag/zoom",
        "why": "The projected reservoir as a turnable solid — the same xyz the triptych shows, live; the lower-left x/y/z gnomon tracks orientation as you rotate.",
        "svg": canvas, "script": script,
        "legend": '<span><i class="a"></i> dense core (accent)</span>'
                  '<span><i class="f"></i> cloud (faint)</span>'
                  '<span><i class="f"></i> sparse tail (fainter)</span>'
                  '<span><i class="dash"></i> x/y/z origin gnomon</span>',
        "reveal": "<b>Reveals:</b> the occupied volume from any angle — rotate and zoom; the gnomon keeps you oriented as the cloud turns.",
        "cls": "fig-mid",
    }
