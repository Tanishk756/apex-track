/* ==========================================================================
   APEX-Track C4ISR Tactical Command Dashboard Engine
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  initHUDCanvas();
  initRadarCanvas();
  startTelemetryLoop();
});

// 1. HUD Canvas Simulator & Renderer
function initHUDCanvas() {
  const canvas = document.getElementById('hudCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  let angle = 0;
  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Reticle Center Overlay
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, 25, 0, Math.PI * 2);
    ctx.moveTo(cx - 35, cy); ctx.lineTo(cx - 10, cy);
    ctx.moveTo(cx + 10, cy); ctx.lineTo(cx + 35, cy);
    ctx.moveTo(cx, cy - 35); ctx.lineTo(cx, cy - 10);
    ctx.moveTo(cx, cy + 10); ctx.lineTo(cx, cy + 35);
    ctx.stroke();

    requestAnimationFrame(render);
  }
  render();
}

// 2. Radar Canvas Renderer
function initRadarCanvas() {
  const canvas = document.getElementById('radarCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    const size = Math.min(canvas.parentElement.clientWidth, canvas.parentElement.clientHeight) * 0.85;
    canvas.width = size;
    canvas.height = size;
  }
  window.addEventListener('resize', resize);
  resize();

  let sweepAngle = 0;
  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const radius = Math.min(cx, cy) - 10;

    // Concentric Rings
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
    ctx.lineWidth = 1;
    for (let r of [0.3, 0.6, 0.9]) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius * r, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Cross lines
    ctx.beginPath();
    ctx.moveTo(cx - radius, cy); ctx.lineTo(cx + radius, cy);
    ctx.moveTo(cx, cy - radius); ctx.lineTo(cx, cy + radius);
    ctx.stroke();

    // Sweep Beam
    sweepAngle += 0.03;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, sweepAngle - 0.3, sweepAngle);
    ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
    ctx.fill();

    requestAnimationFrame(render);
  }
  render();
}

// 3. Telemetry Fetching Loop
async function startTelemetryLoop() {
  const tableBody = document.getElementById('targetTableBody');
  const fpsElem = document.getElementById('fpsCounter');
  if (!tableBody) return;

  async function fetchTargets() {
    try {
      const res = await fetch('/api/v1/targets');
      const data = await res.json();

      if (fpsElem) fpsElem.innerText = '30.0 FPS';

      if (data.targets && data.targets.length > 0) {
        tableBody.innerHTML = data.targets.map(t => `
          <tr>
            <td>#${t.track_id}</td>
            <td>${t.class_name.toUpperCase()}</td>
            <td>${(t.confidence * 100).toFixed(0)}%</td>
            <td>${(t.speed_kmh !== undefined && t.speed_kmh !== null) ? Number(t.speed_kmh).toFixed(1) : '0.0'} km/h</td>

            <td><span style="color:#10b981;">CONFIRMED</span></td>
            <td>
              <button class="btn-action" onclick="lockTarget(${t.track_id})">LOCK</button>
              <button class="btn-action" style="background:#f59e0b;" onclick="triggerJamming(${t.track_id})">JAM</button>
              <button class="btn-action" style="background:#ef4444;" onclick="triggerIntercept(${t.track_id})">INTERCEPT</button>
            </td>
          </tr>
        `).join('');
      } else {
        tableBody.innerHTML = `
          <tr>
            <td colspan="6" style="text-align: center; color: #94a3b8; padding: 20px;">
              SCANNING FOR TARGETS (CAMERA STREAM ACTIVE)...
            </td>
          </tr>
        `;
      }
    } catch (e) {
      console.warn('Backend API connection offline');
    }
  }

  setInterval(fetchTargets, 500);
  fetchTargets();
}

async function setVisionMode(mode) {
  try {
    const res = await fetch(`/api/v1/vision/mode?mode=${mode}`, { method: 'POST' });
    const data = await res.json();
    console.log('Vision mode set to:', data.vision_mode);
  } catch (e) {
    console.error('Failed to set vision mode:', e);
  }
}

async function switchRoboflowModel(modelId) {
  try {
    const res = await fetch(`/api/v1/roboflow/model?model_id=${encodeURIComponent(modelId)}`, { method: 'POST' });
    const data = await res.json();
    console.log('Roboflow model switched to:', data.active_roboflow_model);
  } catch (e) {
    console.error('Failed to switch Roboflow model:', e);
  }
}

async function lockTarget(id) {
  try {
    const res = await fetch(`/api/v1/targets/lock?track_id=${id}`, { method: 'POST' });
    const data = await res.json();
    alert(`OPTICAL GIMBAL LOCKED ON TARGET #${id}`);
  } catch (e) {
    console.error('Failed to lock target:', e);
  }
}

async function triggerJamming(id) {
  try {
    const res = await fetch(`/api/v1/countermeasures/jam?target_id=${id}`, { method: 'POST' });
    const data = await res.json();
    alert(`DIRECTIONAL RF JAMMING ACTIVE ON TARGET #${id}`);
  } catch (e) {
    console.error('Failed to trigger RF jamming:', e);
  }
}

async function triggerIntercept(id) {
  try {
    const res = await fetch(`/api/v1/countermeasures/intercept?target_id=${id}`, { method: 'POST' });
    const data = await res.json();
    alert(`KINETIC INTERCEPT ENGAGED ON TARGET #${id}`);
  } catch (e) {
    console.error('Failed to trigger kinetic intercept:', e);
  }
}



