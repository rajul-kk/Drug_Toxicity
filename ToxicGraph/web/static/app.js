// ── globals populated by /api/info ─────────────
let TASK_NAMES = [];
let DS_COLORS  = [];
let APP = {};

// ── view switching ─────────────────────────────
function showView(name, btn){
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.querySelectorAll('#view-toggle .vt-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('view-'+name).classList.add('active');
  if(btn) btn.classList.add('active');
  document.body.classList.toggle('landing-mode', name === 'home');
  document.body.classList.toggle('app-mode', name !== 'home');
  document.getElementById('nav-query-bar').style.display = name==='home' ? 'none' : '';
  if(name==='browse' && APP.taskNames) fetchBrowse();
}

// ── task tab switching ─────────────────────────
let chartRendered = false;
function switchTaskTab(name, btn){
  document.querySelectorAll('.tp-tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tp-body').forEach(b=>b.classList.add('hidden'));
  btn.classList.add('active');
  document.getElementById('tp-'+name).classList.remove('hidden');
  if(name==='chart' && !chartRendered){
    renderChart(window._isTestSet || false, window._currentGT || null);
    chartRendered = true;
  }
}

// ── plotly chart ───────────────────────────────
function renderChart(isTestSet, gt){
  const means = window._currentMeans || [];
  const stds  = window._currentStds  || [];
  if(!means.length) return;

  const colors = means.map((_,i) => (DS_COLORS[i] || '#64748b') + 'cc');
  const borders = means.map((_,i) => DS_COLORS[i] || '#64748b');
  const modelLabel = (APP.models && APP.models[0]) ? APP.models[0].toUpperCase() : 'Model';

  const traces = [{
    type:'bar', x:TASK_NAMES, y:means,
    error_y:{type:'data',array:stds,visible:true,color:'#9b9590',thickness:1.2,width:3},
    marker:{color:colors, line:{color:borders,width:1}},
    hovertemplate:'<b>%{x}</b><br>prob: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>',
    name: modelLabel,
  }];

  if(window._compareMode && window._currentMeansB){
    const meansB = window._currentMeansB;
    const stdsB  = window._currentStdsB || meansB.map(()=>0);
    const labelB = (APP.models && APP.models[1]) ? APP.models[1].toUpperCase() : 'Model B';
    traces.push({
      type:'bar', x:TASK_NAMES, y:meansB,
      error_y:{type:'data',array:stdsB,visible:true,color:'#b45309',thickness:1.2,width:3},
      marker:{color:meansB.map(()=>'rgba(180,83,9,.35)'), line:{color:meansB.map(()=>'#b45309'),width:1}},
      hovertemplate:'<b>%{x}</b><br>'+labelB+': %{y:.3f} ± %{error_y.array:.3f}<extra></extra>',
      name: labelB,
    });
  }

  if(isTestSet && gt){
    const gtX_pos=[], gtX_neg=[], gtX_mis=[];
    const gtY_pos=[], gtY_neg=[], gtY_mis=[];
    gt.forEach((v,i)=>{
      if(v===1){gtX_pos.push(TASK_NAMES[i]);gtY_pos.push(1.02)}
      else if(v===0){gtX_neg.push(TASK_NAMES[i]);gtY_neg.push(1.02)}
      else{gtX_mis.push(TASK_NAMES[i]);gtY_mis.push(1.02)}
    });
    traces.push({type:'scatter',mode:'markers',x:gtX_pos,y:gtY_pos,name:'GT positive',
      marker:{symbol:'diamond',size:8,color:'#dc2626'},hovertemplate:'<b>%{x}</b><br>GT: positive<extra></extra>'});
    traces.push({type:'scatter',mode:'markers',x:gtX_neg,y:gtY_neg,name:'GT negative',
      marker:{symbol:'circle',size:7,color:'#9b9590'},hovertemplate:'<b>%{x}</b><br>GT: negative<extra></extra>'});
    traces.push({type:'scatter',mode:'markers',x:gtX_mis,y:gtY_mis,name:'Missing',
      marker:{symbol:'x',size:7,color:'#b45309'},hovertemplate:'<b>%{x}</b><br>GT: missing<extra></extra>'});
  }

  const shapes = [], annotations = [];
  if(APP.taskGroups && APP.dsColors){
    let offset = -0.5;
    Object.entries(APP.taskGroups).forEach(([ds, tasks]) => {
      const end = offset + tasks.length;
      const col = APP.dsColors[ds] || '#64748b';
      shapes.push({type:'rect',x0:offset,x1:end,y0:0,y1:1.08,
        fillcolor:col+'08',line:{width:0},layer:'below'});
      annotations.push({x:(offset+end)/2,y:1.06,xref:'x',yref:'y',text:ds,showarrow:false,
        font:{size:10,color:col}});
      offset = end;
    });
  }
  shapes.push({type:'line',x0:-0.5,x1:TASK_NAMES.length-0.5,y0:.5,y1:.5,
    line:{color:'rgba(0,0,0,.1)',width:1,dash:'dot'}});

  const layout = {
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    margin:{t:10,b:80,l:30,r:10},
    height:300,
    barmode: window._compareMode ? 'group' : 'relative',
    xaxis:{tickangle:-45,tickfont:{size:9,family:'JetBrains Mono',color:'#9b9590'},
           gridcolor:'rgba(0,0,0,.05)',zeroline:false},
    yaxis:{range:[0,1.1],gridcolor:'rgba(0,0,0,.06)',zeroline:false,
           tickfont:{size:10,family:'JetBrains Mono',color:'#9b9590'}},
    shapes, annotations,
    legend:{orientation:'h',y:-0.35,font:{size:10,family:'DM Sans'}},
    showlegend: isTestSet || window._compareMode,
    hovermode:'closest',
    font:{family:'DM Sans'},
  };
  Plotly.newPlot('task-chart', traces, layout, {responsive:true,displayModeBar:false});
}

// ── browse state + fetch ───────────────────────
let browseState = {filter:'all', sort:'conf', page:1};

async function fetchBrowse(){
  const tbody = document.getElementById('mol-tbody');
  tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--text-3);font-size:12px">Loading…</td></tr>';
  try{
    const {filter, sort, page} = browseState;
    const url = `/api/testset?model=${APP.activeModel||''}&page=${page}&per_page=20&filter=${filter}&sort=${sort}`;
    const r = await fetch(url);
    if(!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    renderBrowseRows(data.rows);
    updatePagination(data.page, data.pages, data.total);
    document.getElementById('browse-sub').textContent =
      `${data.total} molecules · click any row to load in viewer`;
  } catch(e){
    console.error('fetchBrowse failed:', e);
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--red);font-size:12px">Failed to load: ${e.message}</td></tr>`;
    document.getElementById('browse-sub').textContent = 'Error loading test set';
  }
}

function renderBrowseRows(rows){
  const tbody = document.getElementById('mol-tbody');
  tbody.innerHTML = '';
  rows.forEach(r => {
    const conf = r.max_conf;
    const col = conf>=0.8 ? 'var(--green)' : conf>=0.5 ? 'var(--amber)' : 'var(--blue-lt)';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="thumb-cell">
        <img class="thumb-img"
          src="/api/thumbnail/${encodeURIComponent(r.smiles)}?size=80"
          onerror="this.style.display='none'" alt="" loading="lazy">
      </td>
      <td class="smiles-cell">${r.smiles}</td>
      <td><span class="ds-pill ${r.dataset}">${r.dataset}</span></td>
      <td>
        <div class="conf-bar-wrap">
          <div class="conf-track"><div class="conf-fill" style="width:${conf*100}%;background:${col}"></div></div>
          <span class="conf-val" style="color:${col}">${conf.toFixed(2)}</span>
        </div>
      </td>
      <td><span class="badge ${r.score>=0.75?'g':r.score>=0.55?'b':'a'}">${r.score.toFixed(2)}</span></td>
      <td><button class="open-btn" onclick="openFromBrowse(${r.idx})">Open →</button></td>`;
    tbody.appendChild(tr);
  });
}

function updatePagination(page, pages, total){
  document.getElementById('page-info').textContent =
    `Showing ${(page-1)*20+1}–${Math.min(page*20,total)} of ${total}`;
  const btns = document.getElementById('page-btns');
  btns.innerHTML = '';
  const makeBtn = (label, p) => {
    const b = document.createElement('div');
    b.className = 'page-btn' + (p===page?' active':'');
    b.textContent = label;
    if(p && p!==page) b.onclick = () => { browseState.page=p; fetchBrowse(); };
    btns.appendChild(b);
  };
  if(pages<=7){
    for(let i=1;i<=pages;i++) makeBtn(i,i);
  } else {
    makeBtn(1,1);
    if(page>3) makeBtn('…',null);
    for(let i=Math.max(2,page-1);i<=Math.min(pages-1,page+1);i++) makeBtn(i,i);
    if(page<pages-2) makeBtn('…',null);
    makeBtn(pages,pages);
  }
}

function setFilter(name, el){
  document.querySelectorAll('.fchip').forEach(c=>c.classList.remove('active'));
  el.classList.add('active');
  browseState.filter = name;
  browseState.page = 1;
  fetchBrowse();
}

function sortTable(key){
  browseState.sort = key;
  browseState.page = 1;
  fetchBrowse();
}

async function openFromBrowse(idx){
  const predictBtn = document.querySelector('#view-toggle .vt-btn');
  showView('predict', predictBtn);

  document.getElementById('mol-placeholder').style.display = 'none';
  document.getElementById('predict-loader').style.display = 'flex';
  document.getElementById('predict-loader').innerHTML = '<div class="loader-ring"></div>loading…';

  let data;
  try{
    const r = await fetch(`/api/testset/${idx}?model=${APP.activeModel||''}`);
    if(!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch(e){
    console.error('openFromBrowse failed:', e);
    document.getElementById('predict-loader').innerHTML =
      `<span style="color:var(--red);font-size:11px">⚠ ${e.message}</span>`;
    return;
  }

  document.getElementById('mol-smiles-display').textContent = data.smiles;
  document.getElementById('mol-name-display').textContent =
    `Test set · ${data.dataset} · ${data.model_used.toUpperCase()} · drag to rotate`;
  document.getElementById('mol-mode-badge').textContent = 'Test set';
  document.getElementById('mol-mode-badge').className = 'mch-badge test';
  document.getElementById('gt-legend').style.display = 'flex';
  document.getElementById('new-mol-note').style.display = 'none';

  window._currentMeans = data.probs;
  window._currentStds  = data.probs.map(() => 0);
  window._currentGT    = data.labels;
  window._isTestSet    = true;
  chartRendered = false;
  if(!document.getElementById('tp-chart').classList.contains('hidden')){
    renderChart(true, data.labels);
    chartRendered = true;
  }

  updateSummaryBars(data.probs);

  if(data.sdf){
    document.getElementById('predict-loader').style.display = 'none';
    init3dViewer(document.getElementById('predict-3d'), data.sdf);
  } else {
    document.getElementById('predict-loader').innerHTML =
      '<span style="color:#9b9590;font-size:11px">3D unavailable</span>';
  }
}

// ── summary bars helper ────────────────────────
function updateSummaryBars(means){
  const top6 = [...means.map((v,i)=>({v,i}))]
    .sort((a,b)=>b.v-a.v).slice(0,6);
  const barsHtml = top6.map(({v,i}) => {
    const col = v>=0.7 ? 'var(--green)' : v>=0.4 ? 'var(--blue-lt)' : 'var(--text-3)';
    return `<div class="task-row">
      <span class="task-name">${TASK_NAMES[i]||i}</span>
      <div class="task-track"><div class="task-fill" style="width:${v*100}%;background:${col}"></div></div>
      <span class="task-val" style="color:${col}">${v.toFixed(2)}</span>
    </div>`;
  }).join('');
  document.getElementById('task-bars-container').innerHTML = barsHtml;
}

// ── dual-model prediction helper ───────────────
async function fetchPrediction(smiles, model){
  const r = await fetch('/api/predict', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({smiles, n_mc:20, model}),
  });
  if(!r.ok){ const e=await r.json().catch(()=>({})); throw new Error(e.detail||`HTTP ${r.status}`); }
  return r.json();
}

// ── predict ────────────────────────────────────
async function runPredict(){
  const smiles = document.getElementById('smiles-inp').value.trim();
  if(!smiles) return;

  if(!document.getElementById('view-predict').classList.contains('active')){
    showView('predict', document.querySelector('#view-toggle .vt-btn'));
  }

  document.getElementById('mol-placeholder').style.display = 'none';
  document.getElementById('predict-loader').style.display = 'flex';
  document.getElementById('predict-loader').innerHTML =
    '<div class="loader-ring"></div>predicting…';
  document.getElementById('mol-mode-badge').textContent = 'Predicting…';

  let data, dataB = null;
  const compareMode = APP.models && APP.models.length > 1;
  try{
    if(compareMode){
      [data, dataB] = await Promise.all(
        APP.models.map(m => fetchPrediction(smiles, m))
      );
    } else {
      data = await fetchPrediction(smiles, APP.activeModel||undefined);
    }
  } catch(e){
    document.getElementById('predict-loader').innerHTML =
      `<span style="color:#dc2626;font-size:11px">⚠ ${e.message}</span>`;
    document.getElementById('mol-mode-badge').textContent = 'Error';
    return;
  }

  document.getElementById('mol-smiles-display').textContent = smiles;
  const modelLabel = compareMode
    ? APP.models.map(m=>m.toUpperCase()).join(' vs ')
    : data.model_used.toUpperCase();
  document.getElementById('mol-name-display').textContent =
    `New prediction · ${modelLabel} · click and drag to rotate`;
  document.getElementById('mol-mode-badge').textContent = compareMode ? 'Comparing models' : 'New prediction';
  document.getElementById('mol-mode-badge').className = 'mch-badge new';
  document.getElementById('gt-legend').style.display = 'none';
  document.getElementById('new-mol-note').style.display = 'flex';

  document.getElementById('chip-max-auc').textContent = data.max_auc.toFixed(2);
  document.getElementById('chip-mc-std').textContent = data.mc_std_mean.toFixed(3);
  document.getElementById('chip-top-task').textContent = data.top_task;

  window._currentMeans  = data.means;
  window._currentStds   = data.stds;
  window._currentMeansB = dataB ? dataB.means : null;
  window._currentStdsB  = dataB ? dataB.stds  : null;
  window._currentGT     = null;
  window._isTestSet     = false;
  window._compareMode   = compareMode;
  chartRendered = false;
  if(!document.getElementById('tp-chart').classList.contains('hidden')){
    renderChart(false, null);
    chartRendered = true;
  }

  updateSummaryBars(data.means);

  if(data.sdf){
    document.getElementById('predict-loader').style.display = 'none';
    init3dViewer(document.getElementById('predict-3d'), data.sdf);
  } else {
    document.getElementById('predict-loader').innerHTML =
      '<span style="color:#9b9590;font-size:11px">3D unavailable</span>';
  }
}

// ── model toggle ───────────────────────────────
function setModel(m, btn){
  APP.activeModel = m;
  document.querySelectorAll('#model-toggle .vt-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  if(document.getElementById('view-predict').classList.contains('active')){
    const smiles = document.getElementById('smiles-inp').value.trim();
    if(smiles) runPredict();
  } else {
    fetchBrowse();
  }
}

// ── boot ───────────────────────────────────────
async function boot(){
  try{
    const r = await fetch('/api/info');
    if(!r.ok) throw new Error(`/api/info returned HTTP ${r.status}`);
    const info = await r.json();

    TASK_NAMES = info.task_names;
    APP.taskNames   = info.task_names;
    APP.taskGroups  = info.task_groups;
    APP.dsColors    = info.dataset_colors;
    APP.models      = info.available_models;
    APP.activeModel = info.default_model;

    DS_COLORS = [];
    for(const [ds, tasks] of Object.entries(info.task_groups)){
      const col = info.dataset_colors[ds] || '#64748b';
      tasks.forEach(() => DS_COLORS.push(col));
    }

    const toggle = document.getElementById('model-toggle');
    if(info.available_models.length > 1){
      toggle.innerHTML = info.available_models.map(m =>
        `<button class="vt-btn ${m===APP.activeModel?'active':''}"
           onclick="setModel('${m}',this)">${m.toUpperCase()}</button>`
      ).join('');
      toggle.style.display = 'flex';
    }

    const chips = document.getElementById('filter-chips');
    chips.innerHTML = `<div class="fchip active all" onclick="setFilter('all',this)">All</div>` +
      Object.keys(info.task_groups).map(ds =>
        `<div class="fchip ${ds}" onclick="setFilter('${ds}',this)">${ds}</div>`
      ).join('');

    document.getElementById('hs-tasks').textContent = info.task_names.length;
    document.getElementById('hs-datasets').textContent = Object.keys(info.task_groups).length;
    document.getElementById('hs-models').textContent = '×' + info.available_models.length;
    document.getElementById('hero-eyebrow').textContent =
      info.available_models.map(m=>m.toUpperCase()).join(' + ') +
      ' · ' + Object.keys(info.task_groups).length + ' datasets · ' +
      info.task_names.length + ' tasks';

    fetch('/api/testset/0?model=' + APP.activeModel)
      .then(r=>r.ok ? r.json() : null)
      .then(d=>{
        const sdf = d && d.sdf;
        const el = document.getElementById('hero3d');
        document.getElementById('hero-loader').style.display = 'none';
        if(sdf && el){ init3dViewer(el, sdf, 0.3); }
      }).catch(()=>{ document.getElementById('hero-loader').style.display='none'; });

    fetchBrowse();
  } catch(e){
    console.error('boot failed:', e);
    document.getElementById('browse-sub').textContent = 'Error: ' + e.message;
    document.getElementById('mol-tbody').innerHTML =
      `<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--red);font-size:12px">App failed to initialise: ${e.message}</td></tr>`;
  }
}

boot();
