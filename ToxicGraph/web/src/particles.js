(function(){
  const canvas = document.getElementById('bg-canvas');
  const ctx    = canvas.getContext('2d');

  /* three depth layers */
  const CFG = {
    bokeh: { count: 9,  rMin: 46, rMax: 94,  speed: .07,  breathAmp: .07,  breathFreq: 3.6  },
    mid:   { count: 50, rMin: 1.4, rMax: 3.2, speed: .18,  breathAmp: .13,  breathFreq: 5.2  },
    back:  { count: 80, rMin: .4,  rMax: 1.1, speed: .28 },
    linkDist: 148,
  };

  let W, H, t = 0;
  let layers = { bokeh: [], mid: [], back: [] };

  function resize() { W = canvas.width = innerWidth; H = canvas.height = innerHeight; }

  function mk(cfg, extra) {
    const ang = Math.random() * Math.PI * 2;
    const spd = cfg.speed * (.4 + Math.random() * .6);
    return Object.assign({
      x:     Math.random() * W,
      y:     Math.random() * H,
      r:     cfg.rMin + Math.random() * (cfg.rMax - cfg.rMin),
      vx:    Math.cos(ang) * spd,
      vy:    Math.sin(ang) * spd,
      phase: Math.random() * Math.PI * 2,
    }, extra);
  }

  function wrap(p, pad) {
    if (p.x < -pad)    p.x = W + pad;
    if (p.x > W + pad) p.x = -pad;
    if (p.y < -pad)    p.y = H + pad;
    if (p.y > H + pad) p.y = -pad;
  }

  function tick() {
    /* t advances ~0.24 rad/s at 60 fps (0.004 * 60) */
    t += .004;
    ctx.clearRect(0, 0, W, H);

    /* ── layer 0: distant specks ─────────────────────────── */
    ctx.fillStyle = 'rgba(255,255,255,.16)';
    for (const p of layers.back) {
      p.x += p.vx; p.y += p.vy; wrap(p, 8);
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    /* ── layer 1: midfield connected graph ───────────────── */
    const m = layers.mid;

    /* edges */
    for (let i = 0; i < m.length; i++) {
      for (let j = i + 1; j < m.length; j++) {
        const a = m[i], b = m[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < CFG.linkDist) {
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(255,255,255,${(.12 * (1 - d / CFG.linkDist)).toFixed(3)})`;
          ctx.lineWidth   = .55;
          ctx.stroke();
        }
      }
    }

    /* nodes */
    for (const p of m) {
      p.x += p.vx; p.y += p.vy; wrap(p, 24);
      const s = 1 + CFG.mid.breathAmp * Math.sin(t * CFG.mid.breathFreq + p.phase);
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * s, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${(.45 + .3 * p.depth).toFixed(2)})`;
      ctx.fill();
    }

    /* ── layer 2: foreground bokeh orbs ──────────────────── */
    for (const p of layers.bokeh) {
      p.x += p.vx; p.y += p.vy; wrap(p, p.r + 20);
      const s = 1 + CFG.bokeh.breathAmp * Math.sin(t * CFG.bokeh.breathFreq + p.phase);
      const r = p.r * s;

      /* soft radial gradient — centre bright, edge fully transparent */
      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r);
      const a = p.baseAlpha;
      g.addColorStop(0,    `rgba(255,255,255,${(a * 2.4).toFixed(3)})`);
      g.addColorStop(.28,  `rgba(255,255,255,${(a * 1.5).toFixed(3)})`);
      g.addColorStop(.62,  `rgba(255,255,255,${(a * .55).toFixed(3)})`);
      g.addColorStop(1,    'rgba(255,255,255,0)');
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = g;
      ctx.fill();
    }

    if (!document.hidden) requestAnimationFrame(tick);
  }

  function init() {
    resize();
    layers.back  = Array.from({ length: CFG.back.count  }, () => mk(CFG.back));
    layers.mid   = Array.from({ length: CFG.mid.count   }, () => mk(CFG.mid,  {
      depth: Math.random(),   /* used for slight brightness variation */
    }));
    layers.bokeh = Array.from({ length: CFG.bokeh.count }, () => mk(CFG.bokeh, {
      baseAlpha: .026 + Math.random() * .034,
    }));
    tick();
  }

  window.addEventListener('resize', () => resize());
  document.addEventListener('visibilitychange', () => { if (!document.hidden) requestAnimationFrame(tick); });

  init();
})();
