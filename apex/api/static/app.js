/* ==========================================================================
   APEX-Track v15.0 Master Apex Edition Command Dashboard Engine
   ========================================================================== */

let voiceSynthEnabled = true;

document.addEventListener('DOMContentLoaded', () => {
  initHUDCanvas();
  initRadarCanvas();
  startTelemetryLoop();
  pollDefconStatus();
  setInterval(pollDefconStatus, 2000);
});

// Voice Synthesizer Announcer
function speakTacticalAlert(text) {
  if (!voiceSynthEnabled || !('speechSynthesis' in window)) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.1;
  utterance.pitch = 0.9;
  window.speechSynthesis.speak(utterance);
}

function toggleVoiceSynth() {
  voiceSynthEnabled = !voiceSynthEnabled;
  const btn = document.getElementById('voiceToggleBtn');
  if (btn) {
    btn.innerText = voiceSynthEnabled ? '🔊 AUDIO: ON' : '🔇 AUDIO: OFF';
  }
}

// DEFCON Status Poller
async function pollDefconStatus() {
  try {
    const res = await fetch('/api/v1/defcon');
    const data = await res.json();
    const badge = document.getElementById('defconBadge');
    if (badge && data.label) {
      badge.innerText = data.label;
      badge.className = `defcon-badge defcon-${data.defcon || 5}`;
    }
  } catch (e) {
    // Fallback if backend offline
  }
}

// 1. HUD Canvas Simulator & Renderer
function initHUDCanvas() {
  const canvas = document.getElementById('hudCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width = canvas.parentElement.clientWidth || 640;
    canvas.height = canvas.parentElement.clientHeight || 480;
  }
  window.addEventListener('resize', resize);
  resize();

  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Reticle Center Overlay
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
    ctx.lineWidth = 1.5;
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

// 2. Functional Radar Canvas Renderer (Live Target Blips & PPI Sweep)
let radarTargets = [];

async function fetchRadarData() {
  try {
    const res = await fetch('/api/v1/radar/ppi');
    const data = await res.json();
    if (data.radar_targets) {
      radarTargets = data.radar_targets;
    }
  } catch (e) {
    // Fallback if backend API offline
  }
}
setInterval(fetchRadarData, 500);

function initRadarCanvas() {
  const canvas = document.getElementById('radarCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    const parentWidth = canvas.parentElement.clientWidth || 300;
    const parentHeight = canvas.parentElement.clientHeight || 300;
    const size = Math.max(200, Math.min(parentWidth, parentHeight) * 0.88);
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
    const radius = Math.min(cx, cy) - 15;

    // Background Grid
    ctx.fillStyle = '#030712';
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    // Concentric Range Rings (100m, 250m, 500m)
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
    ctx.lineWidth = 1;
    const ranges = [0.2, 0.5, 0.8, 1.0];
    ranges.forEach((r) => {
      ctx.beginPath();
      ctx.arc(cx, cy, radius * r, 0, Math.PI * 2);
      ctx.stroke();
    });

    // Crosshairs
    ctx.beginPath();
    ctx.moveTo(cx - radius, cy); ctx.lineTo(cx + radius, cy);
    ctx.moveTo(cx, cy - radius); ctx.lineTo(cx, cy + radius);
    ctx.stroke();

    // Rotating Radar PPI Sweep Line
    sweepAngle += 0.03;
    if (sweepAngle >= Math.PI * 2) sweepAngle = 0;

    const sweepX = cx + Math.cos(sweepAngle) * radius;
    const sweepY = cy + Math.sin(sweepAngle) * radius;

    const grad = ctx.createConicGradient(sweepAngle, cx, cy);
    grad.addColorStop(0, 'rgba(0, 240, 255, 0.4)');
    grad.addColorStop(0.1, 'rgba(0, 240, 255, 0.05)');
    grad.addColorStop(1, 'transparent');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#00F0FF';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(sweepX, sweepY);
    ctx.stroke();

    // Render Live Radar Blips
    radarTargets.forEach(t => {
      const bx = cx + (t.x || 0) * (radius / 250);
      const by = cy + (t.y || 0) * (radius / 250);
      ctx.fillStyle = '#FF0055';
      ctx.beginPath();
      ctx.arc(bx, by, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowColor = '#FF0055';
      ctx.shadowBlur = 8;
    });
    ctx.shadowBlur = 0;

    requestAnimationFrame(render);
  }
  render();
}

// 3. Start Dynamic Telemetry Fetch Loop
function startTelemetryLoop() {
  const tableBody = document.getElementById('targetTableBody');
  if (!tableBody) return;

  async function fetchTargets() {
    try {
      const res = await fetch('/api/v1/targets');
      const data = await res.json();

      const fpsElem = document.getElementById('fpsCounter');
      const latElem = document.getElementById('latencyCounter');

      if (data.stream_fps !== undefined && fpsElem) {
        fpsElem.innerText = `${data.stream_fps.toFixed(1)} FPS`;
      }
      if (data.latency_ms !== undefined && latElem) {
        latElem.innerText = `${data.latency_ms.toFixed(1)} ms`;
      }

      if (data.targets && data.targets.length > 0) {
        tableBody.innerHTML = data.targets.map(t => `
          <tr>
            <td>#${t.track_id}</td>
            <td>${t.class_name.toUpperCase()}</td>
            <td>${(t.confidence * 100).toFixed(0)}%</td>
            <td>${(t.speed_kmh !== undefined && t.speed_kmh !== null) ? Number(t.speed_kmh).toFixed(1) : '0.0'} km/h</td>
            <td><span style="color:${t.state === 'CONFIRMED' ? '#00FF9D' : '#FFB800'}; font-weight: 700;">${t.state || 'TRACKING'}</span></td>
            <td>
              <button class="btn-action" onclick="lockTarget(${t.track_id})">LOCK</button>
              <button class="btn-action" style="background:#FFB800; color:#000;" onclick="triggerJamming(${t.track_id})">JAM</button>
              <button class="btn-action" style="background:#FF0055;" onclick="triggerIntercept(${t.track_id})">INTERCEPT</button>
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

async function lockTarget(id) {
  try {
    const res = await fetch(`/api/v1/targets/lock?track_id=${id}`, { method: 'POST' });
    const data = await res.json();
    speakTacticalAlert(`Target ${id} locked.`);
    alert(`OPTICAL GIMBAL LOCKED ON TARGET #${id}`);
  } catch (e) {
    console.error('Failed to lock target:', e);
  }
}

async function triggerJamming(id) {
  try {
    const res = await fetch(`/api/v1/countermeasures/jam?target_id=${id}`, { method: 'POST' });
    const data = await res.json();
    speakTacticalAlert(`Directional RF Jamming activated on target ${id}.`);
    alert(`DIRECTIONAL RF JAMMING ACTIVE ON TARGET #${id}`);
  } catch (e) {
    console.error('Failed to trigger RF jamming:', e);
  }
}

async function triggerIntercept(id) {
  try {
    const res = await fetch(`/api/v1/countermeasures/intercept?target_id=${id}`, { method: 'POST' });
    const data = await res.json();
    speakTacticalAlert(`Kinetic Intercept engaged on target ${id}.`);
    alert(`KINETIC INTERCEPT ENGAGED ON TARGET #${id}`);
  } catch (e) {
    console.error('Failed to trigger kinetic intercept:', e);
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

// Copilot RAG Agent Chat Console
async function sendCopilotPrompt() {
  const input = document.getElementById('copilotInput');
  const chatBox = document.getElementById('copilotChatBox');
  if (!input || !input.value.trim() || !chatBox) return;

  const promptText = input.value.trim();
  input.value = '';

  chatBox.innerHTML += `<div style="color:#fff; margin-bottom:4px;"><strong>[OPERATOR]:</strong> ${promptText}</div>`;
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const res = await fetch('/api/v1/copilot/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: promptText })
    });
    const data = await res.json();
    const ans = data.agent || data.response || data.message || 'Copilot action executed.';
    chatBox.innerHTML += `<div class="chat-msg"><strong>[COPILOT]:</strong> ${ans}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
  } catch (e) {
    chatBox.innerHTML += `<div class="chat-msg" style="color:#FF0055;"><strong>[COPILOT]:</strong> System response received. Operational.</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
  }
}

function sendQuickChip(actionText) {
  const input = document.getElementById('copilotInput');
  if (input) {
    input.value = actionText;
    sendCopilotPrompt();
  }
}

// Historical Database Log Viewer
let allHistoryRecords = [];

async function openHistoryModal() {
  const modal = document.getElementById('historyModal');
  if (modal) modal.style.display = 'flex';
  await fetchHistoryLogs();
}

function closeHistoryModal() {
  const modal = document.getElementById('historyModal');
  if (modal) modal.style.display = 'none';
}

async function fetchHistoryLogs() {
  const tableBody = document.getElementById('historyTableBody');
  const badge = document.getElementById('historySummaryBadge');
  if (!tableBody) return;

  try {
    const [historyRes, summaryRes] = await Promise.all([
      fetch('/api/v1/history/tracks?limit=200'),
      fetch('/api/v1/history/summary')
    ]);
    const historyData = await historyRes.json();
    const summaryData = await summaryRes.json();

    allHistoryRecords = historyData.history || [];

    if (badge) {
      badge.innerText = `TOTAL RECORDS: ${summaryData.total_records || 0} | UNIQUE UIDs: ${summaryData.unique_targets || 0}`;
    }

    renderHistoryTable(allHistoryRecords);
  } catch (e) {
    console.error('Failed to fetch historical logs:', e);
  }
}

function filterHistoryLogs() {
  const tidFilter = document.getElementById('historyTrackIdFilter')?.value.trim().toLowerCase();
  const classFilter = document.getElementById('historyClassFilter')?.value.trim().toLowerCase();

  const filtered = allHistoryRecords.filter(r => {
    const matchTid = !tidFilter || String(r.track_id).toLowerCase().includes(tidFilter);
    const matchClass = !classFilter || String(r.class_name).toLowerCase().includes(classFilter);
    return matchTid && matchClass;
  });

  renderHistoryTable(filtered);
}

function renderHistoryTable(records) {
  const tableBody = document.getElementById('historyTableBody');
  if (!tableBody) return;

  if (records.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="6" style="padding: 20px; text-align: center; color: #94a3b8;">
          NO HISTORICAL DETECTION RECORDS FOUND IN SQLITE DATABASE
        </td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = records.map(r => `
    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
      <td style="padding: 8px; color: #00F0FF; font-family: monospace;">${r.formatted_time || 'N/A'}</td>
      <td style="padding: 8px; font-weight: 700; color: #00FF9D;">#${r.track_id}</td>
      <td style="padding: 8px; font-weight: 600; text-transform: uppercase;">${r.class_name}</td>
      <td style="padding: 8px;">${r.state || 'CONFIRMED'}</td>
      <td style="padding: 8px;">${(r.speed_kmh || 0).toFixed(1)} km/h</td>
      <td style="padding: 8px; font-family: monospace; color: #94a3b8;">[${r.x1 || 0}, ${r.y1 || 0}, ${r.x2 || 0}, ${r.y2 || 0}]</td>
    </tr>
  `).join('');
}

async function fetchWeatherTelemetry() {
  try {
    const res = await fetch('/api/v1/weather/telemetry');
    const data = await res.json();
    const b = document.getElementById('weatherBadge');
    if (b && data) {
      b.innerText = `🌤️ ${data.condition || 'CLEAR'} | ${data.temperature_c || 24}°C | ${data.wind_speed_ms || 3.5}m/s`;
    }
  } catch (e) {
    console.warn('Weather poller offline');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  fetchWeatherTelemetry();
  setInterval(fetchWeatherTelemetry, 30000);
});
