// ── globals populated by /api/info ─────────────
// var (not let) so these are window properties accessible across all script files
var TASK_NAMES = [];
var DS_COLORS  = [];
var APP = {};
var isPredicting = false;

// ── render-state flags (accessed by predict.js and browse.js) ──
var chartRendered = false;
var tableRendered = false;

// ── view switching ─────────────────────────────
function showView(name, btn){
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.querySelectorAll('#view-toggle .vt-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('view-'+name).classList.add('active');
  if(btn) btn.classList.add('active');
  document.body.classList.toggle('landing-mode', name === 'home');
  document.body.classList.toggle('app-mode', name !== 'home');
  document.getElementById('nav-query-bar').style.display = name==='home' ? 'none' : '';
  const mt = document.getElementById('model-toggle');
  if(mt) mt.style.display = (name === 'predict' && APP.models && APP.models.length > 1) ? '' : 'none';
  if(name === 'home') syncHashToUrl('');
  if(name==='browse' && APP.taskNames) fetchBrowse();
}

// ── task tab switching ─────────────────────────
function switchTaskTab(name, btn){
  document.querySelectorAll('.tp-tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tp-body').forEach(b=>b.classList.add('hidden'));
  btn.classList.add('active');
  document.getElementById('tp-'+name).classList.remove('hidden');
  if(name==='chart' && !chartRendered){
    renderChart(window._isTestSet || false, window._currentGT || null);
    chartRendered = true;
  }
  if(name==='table' && !tableRendered){
    renderTable();
    tableRendered = true;
  }
}

function toggleMissingLabels(){
  chartRendered = false;
  renderChart(window._isTestSet || false, window._currentGT || null);
  chartRendered = true;
}

function copySmiles(el){
  const text = el.textContent.trim();
  if(!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const orig = el.textContent;
    el.textContent = 'Copied!';
    el.style.color = 'var(--green)';
    setTimeout(() => { el.textContent = orig; el.style.color = ''; }, 1200);
  });
}

// ── shareable URL hash ─────────────────────────
function syncHashToUrl(smiles){
  const hash = smiles ? '#predict?s=' + encodeURIComponent(smiles) : '#';
  history.replaceState(null, '', hash);
}

function loadFromHash(){
  const m = location.hash.match(/[#?&]s=([^&]+)/);
  if(!m) return;
  const smiles = decodeURIComponent(m[1]);
  document.getElementById('smiles-inp').value = smiles;
  showView('predict', document.querySelector('#view-toggle .vt-btn'));
  runPredict();
}

// ── model toggle ───────────────────────────────
// m === null means run all available models side-by-side (compare mode)
function setModel(m, btn){
  APP.activeModel = m;
  document.querySelectorAll('#model-toggle .vt-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  const smiles = document.getElementById('smiles-inp').value.trim();
  if(smiles) runPredict();
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
    APP.activeModel = info.available_models.length > 1 ? null : info.default_model;

    // populate model toggle (only rendered when predict view is active)
    const mt = document.getElementById('model-toggle');
    if(mt && APP.models.length > 1){
      mt.innerHTML = APP.models.map(m =>
        `<button class="vt-btn" onclick="setModel('${m}',this)">${m.toUpperCase()}</button>`
      ).join('') + `<button class="vt-btn active" onclick="setModel(null,this)">Both</button>`;
    }

    DS_COLORS = [];
    for(const [ds, tasks] of Object.entries(info.task_groups)){
      const col = info.dataset_colors[ds] || '#64748b';
      tasks.forEach(() => DS_COLORS.push(col));
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
    historyRender();
    loadFromHash();
  } catch(e){
    console.error('boot failed:', e);
    document.getElementById('browse-sub').textContent = 'Error: ' + e.message;
    document.getElementById('mol-tbody').innerHTML =
      `<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--red);font-size:12px">App failed to initialise: ${e.message}</td></tr>`;
  }
}

boot();
