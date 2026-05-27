(function(){
  const canvas = document.getElementById('bg-canvas');
  const ctx    = canvas.getContext('2d');
  const COLORS = ['#6d28d9','#8b5cf6','#2563eb','#1d4ed8'];
  const COUNT  = 60, LINK_DIST = 130, SPEED = 0.4;
  let W, H, particles = [], mouse = {x:-9999,y:-9999};

  function resize(){ W=canvas.width=innerWidth; H=canvas.height=innerHeight; }

  function mkParticle(){
    const a=Math.random()*Math.PI*2;
    return {x:Math.random()*W,y:Math.random()*H,
            vx:Math.cos(a)*SPEED*(0.4+Math.random()*0.6),
            vy:Math.sin(a)*SPEED*(0.4+Math.random()*0.6),
            r:1.5+Math.random()*2,
            color:COLORS[Math.floor(Math.random()*COLORS.length)]};
  }

  function tick(){
    ctx.clearRect(0,0,W,H);
    for(const p of particles){
      const dx=p.x-mouse.x,dy=p.y-mouse.y,d=Math.sqrt(dx*dx+dy*dy);
      if(d<100&&d>0){const f=(100-d)/100*0.5;p.vx+=(dx/d)*f;p.vy+=(dy/d)*f;}
      const spd=Math.sqrt(p.vx*p.vx+p.vy*p.vy);
      if(spd>SPEED*1.8){p.vx=p.vx/spd*SPEED*1.8;p.vy=p.vy/spd*SPEED*1.8;}
      if(spd<0.05){const a=Math.random()*Math.PI*2;p.vx+=Math.cos(a)*.04;p.vy+=Math.sin(a)*.04;}
      p.vx*=.99;p.vy*=.99;
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0){p.x=0;p.vx=Math.abs(p.vx);}
      if(p.x>W){p.x=W;p.vx=-Math.abs(p.vx);}
      if(p.y<0){p.y=0;p.vy=Math.abs(p.vy);}
      if(p.y>H){p.y=H;p.vy=-Math.abs(p.vy);}
    }
    for(let i=0;i<particles.length;i++){
      for(let j=i+1;j<particles.length;j++){
        const a=particles[i],b=particles[j];
        const dx=a.x-b.x,dy=a.y-b.y,d=Math.sqrt(dx*dx+dy*dy);
        if(d<LINK_DIST){
          ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
          ctx.strokeStyle=`rgba(109,40,217,${0.22*(1-d/LINK_DIST)})`;
          ctx.lineWidth=0.7;ctx.stroke();
        }
      }
    }
    for(const p of particles){
      const h=p.color.slice(1);
      const r=parseInt(h.slice(0,2),16),g=parseInt(h.slice(2,4),16),b=parseInt(h.slice(4,6),16);
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(${r},${g},${b},.7)`;ctx.fill();
    }
    if(!document.hidden) requestAnimationFrame(tick);
  }

  window.addEventListener('resize',()=>{resize();});
  window.addEventListener('mousemove',e=>{mouse.x=e.clientX;mouse.y=e.clientY;});
  window.addEventListener('mouseleave',()=>{mouse.x=-9999;mouse.y=-9999;});
  document.addEventListener('visibilitychange',()=>{ if(!document.hidden) requestAnimationFrame(tick); });
  resize();
  particles=Array.from({length:COUNT},mkParticle);
  tick();
})();
