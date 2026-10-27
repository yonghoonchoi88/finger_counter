// 손가락 개수 검사 웹 화면 동작 (ROI 선택 · 설정 · 0.4초마다 상태 갱신)
const $ = id => document.getElementById(id);
const img = $('video');
const canvas = $('selection');
const ctx = canvas.getContext('2d');
let dragging = false, start = null, selected = null, currentTarget = undefined;

// ---------- ROI 선택 (병뚜껑 화면과 동일) ----------
function sizeCanvas() {
  const box = img.getBoundingClientRect();
  const width = Math.round(box.width), height = Math.round(box.height);
  if (width && height && (canvas.width !== width || canvas.height !== height)) {
    canvas.width = width; canvas.height = height; drawSelection();
  }
}
function drawSelection(current = null) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const r = current || selected;
  if (!r) return;
  ctx.fillStyle = 'rgba(255, 200, 45, 0.08)';
  ctx.strokeStyle = '#facc40';
  ctx.lineWidth = 3;
  ctx.setLineDash([7, 4]);
  const x = r.x1 * canvas.width, y = r.y1 * canvas.height;
  const w = (r.x2 - r.x1) * canvas.width, h = (r.y2 - r.y1) * canvas.height;
  ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
}
function pointFromEvent(e) {
  const box = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (e.clientX - box.left) / box.width)),
    y: Math.max(0, Math.min(1, (e.clientY - box.top) / box.height))
  };
}
const rectangle = (a, b) => ({x1:Math.min(a.x,b.x), y1:Math.min(a.y,b.y), x2:Math.max(a.x,b.x), y2:Math.max(a.y,b.y)});

async function requestJson(path, options) {
  const response = await fetch(path, options);
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    const d = result.detail;
    throw new Error(typeof d === 'string' ? d : Array.isArray(d) ? '입력값을 확인하세요.' : '서버 요청 실패');
  }
  return result;
}
const send = (method, path, data) => requestJson(path, {
  method, headers:{'Content-Type':'application/json'}, body: data === undefined ? undefined : JSON.stringify(data)
});

canvas.addEventListener('pointerdown', e => {
  e.preventDefault(); sizeCanvas();
  dragging = true; start = pointFromEvent(e);
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', e => { if (dragging) drawSelection(rectangle(start, pointFromEvent(e))); });
canvas.addEventListener('pointerup', async e => {
  if (!dragging) return;
  dragging = false;
  const rect = rectangle(start, pointFromEvent(e));
  if (rect.x2 - rect.x1 < 0.1 || rect.y2 - rect.y1 < 0.1) {
    $('notice').textContent = 'ROI가 너무 작습니다. 손 전체가 들어가도록 넓게 드래그하세요.';
    drawSelection(); return;
  }
  try {
    await send('POST', '/api/roi', rect);
    selected = rect; drawSelection(); $('notice').textContent = '';
  } catch (err) { $('notice').textContent = err.message; drawSelection(); }
});
canvas.addEventListener('pointercancel', () => { dragging = false; drawSelection(); });

// ---------- 버튼 · 설정 ----------
$('connect').onclick = async () => {
  try {
    await send('POST', '/api/camera', {url: $('camera-url').value.trim()});
    selected = null; drawSelection();
    $('notice').textContent = '카메라 연결을 시도합니다. 영상이 나오면 ROI를 선택하세요.';
  } catch (err) { $('notice').textContent = err.message; }
};
$('reset-roi').onclick = async () => {
  try {
    await send('DELETE', '/api/roi');
    selected = null; drawSelection();
    $('notice').textContent = 'ROI를 초기화했습니다. 다시 드래그하세요.';
  } catch (err) { $('notice').textContent = err.message; }
};
$('mode').onchange = async () => {
  try { await send('POST', '/api/settings', {mode: $('mode').value}); }
  catch (err) { $('notice').textContent = err.message; }
};
$('show-mask').onchange = async () => {
  try { await send('POST', '/api/settings', {show_mask: $('show-mask').checked}); }
  catch (err) { $('notice').textContent = err.message; }
};

// 목표 손가락 수 버튼: 없음, 0 ~ 5
const targetValues = [null, 0, 1, 2, 3, 4, 5];
$('targets').replaceChildren(...targetValues.map(v => {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = v === null ? '없음' : String(v);
  b.dataset.value = v === null ? '' : String(v);
  b.onclick = async () => {
    try { await send('POST', '/api/target', {target: v}); }
    catch (err) { $('notice').textContent = err.message; }
  };
  return b;
}));
function markTarget(target) {
  if (target === currentTarget) return;
  currentTarget = target;
  for (const b of $('targets').children) {
    const active = (b.dataset.value === '' ? null : Number(b.dataset.value)) === target;
    b.classList.toggle('active', active);
    b.setAttribute('aria-pressed', String(active));
  }
}

// ---------- 상태 갱신 ----------
const fmt = (v, digits) => v === null || v === undefined ? '-' : Number(v).toFixed(digits);
async function refreshStatus() {
  try {
    const data = await requestJson('/api/status');
    $('camera-state').textContent = data.camera_status;
    $('fingers').textContent = data.fingers ?? '-';
    $('defect-fingers').textContent = data.status !== 'COUNTED' ? '-'
      : data.defect_fingers === null ? '0~1' : data.defect_fingers;
    $('threshold').textContent = fmt(data.threshold, 1);
    $('area').textContent = fmt(data.area_ratio, 3);
    $('palm').textContent = fmt(data.palm_radius, 1);
    $('mode').value = data.mode;
    $('show-mask').checked = data.show_mask;
    markTarget(data.target);
    if (data.roi && !dragging) {
      selected = {x1:data.roi[0], y1:data.roi[1], x2:data.roi[2], y2:data.roi[3]};
      drawSelection();
    }

    [...$('dots').children].forEach((d, i) => d.classList.toggle('on', data.fingers !== null && i < data.fingers));

    const el = $('decision');
    el.className = '';
    if (data.decision === 'OK') {
      el.textContent = 'OK'; el.className = 'ok';
      $('meaning').textContent = `목표 ${data.target}개와 일치 (손가락 ${data.fingers}개)`;
    } else if (data.decision === 'NG') {
      el.textContent = 'NG'; el.className = 'ng';
      $('meaning').textContent = `목표 ${data.target}개, 현재 ${data.fingers}개`;
    } else if (data.decision === 'COUNTED') {
      el.textContent = `${data.fingers}개`; el.className = 'count';
      $('meaning').textContent = '펴고 있는 손가락 수';
    } else if (data.decision === 'CHECK') {
      el.textContent = '손 없음'; el.className = 'check';
      $('meaning').textContent = 'ROI 안에서 손을 찾지 못했습니다. 이진화 방식을 바꿔 보세요.';
    } else {
      el.textContent = '대기';
      $('meaning').textContent = data.camera_status !== 'ONLINE' ? '카메라 연결 확인 필요' : 'ROI를 선택하세요.';
    }
  } catch (err) {
    $('notice').textContent = '상태 확인 실패: ' + err.message;
  }
}

img.addEventListener('load', sizeCanvas);
new ResizeObserver(sizeCanvas).observe(img);
// ---------- ④ 동작 원리 그림 (데이터 흐름 / 알고리즘 · 한국어 / English) ----------
const how = {view: 'architecture', lang: 'ko'};
const HOW_FILES = {architecture: 'png', pipeline: 'jpg'};
const HOW_ALT = {architecture: '데이터 흐름 그림', pipeline: '알고리즘 단계 그림'};
function showHow() {
  const src = `/assets/${how.view}_${how.lang}.${HOW_FILES[how.view]}`;
  $('how-image').src = src;
  $('how-image').alt = HOW_ALT[how.view];
  $('how-link').href = src;
  document.querySelectorAll('#how-it-works [data-view]').forEach(b => {
    const on = b.dataset.view === how.view;
    b.classList.toggle('active', on); b.setAttribute('aria-selected', String(on));
  });
  document.querySelectorAll('#how-it-works [data-lang]').forEach(b => {
    const on = b.dataset.lang === how.lang;
    b.classList.toggle('active', on); b.setAttribute('aria-pressed', String(on));
  });
}
document.querySelectorAll('#how-it-works [data-view]').forEach(b => b.onclick = () => { how.view = b.dataset.view; showHow(); });
document.querySelectorAll('#how-it-works [data-lang]').forEach(b => b.onclick = () => { how.lang = b.dataset.lang; showHow(); });

setInterval(refreshStatus, 400);
refreshStatus(); sizeCanvas();
