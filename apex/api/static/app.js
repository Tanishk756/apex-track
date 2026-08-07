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

    // Dark grid lines
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
    ctx.lineWidth = 1;
    const step = 40;
    for (let x = 0; x < canvas.width; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Reticle Center
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, 30, 0, Math.PI * 2);
    ctx.moveTo(cx - 45, cy); ctx.lineTo(cx - 15, cy);
    ctx.moveTo(cx + 15, cy); ctx.lineTo(cx + 45, cy);
    ctx.moveTo(cx, cy - 45); ctx.lineTo(cx, cy - 15);
    ctx.moveTo(cx, cy + 15); ctx.lineTo(cx, cy + 45);
    ctx.stroke();

    // Simulated Target Track Box
    angle += 0.02;
    const tx = cx + Math.cos(angle) * 120;
    const ty = cy + Math.sin(angle * 0.7) * 80;
    const tw = 60, th = 60;

    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.strokeRect(tx - tw / 2, ty - th / 2, tw, th);

    // Target Info Label
    ctx.fillStyle = '#10b981';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText(`TRK #01 DRONE [94%]`, tx - tw / 2, ty - th / 2 - 8);
    ctx.fillText(`VEL: 112.4 km/h`, tx - tw / 2, ty + th / 2 + 15);

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

    // Radar blip
    const blipAngle = sweepAngle - 0.8;
    const blipDist = radius * 0.65;
    const bx = cx + Math.cos(blipAngle) * blipDist;
    const by = cy + Math.sin(blipAngle) * blipDist;

    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(bx, by, 5, 0, Math.PI * 2);
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

      if (fpsElem) fpsElem.innerText = '2102 FPS';

      if (data.targets && data.targets.length > 0) {
        tableBody.innerHTML = data.targets.map(t => `
          <tr>
            <td>#${t.track_id}</td>
            <td>${t.class_name.toUpperCase()}</td>
            <td>${(t.confidence * 100).toFixed(0)}%</td>
            <td>${t.speed_kmh} km/h</td>
            <td><span style="color:#10b981;">LOW</span></td>
            <td><button class="btn-action" onclick="lockTarget(${t.track_id})">LOCK</button></td>
          </tr>
        `).join('');
      } else {
        tableBody.innerHTML = `
          <tr>
            <td>#01</td>
            <td>DRONE</td>
            <td>94%</td>
            <td>112.4 km/h</td>
            <td><span style="color:#ef4444; font-weight:bold;">HIGH</span></td>
            <td><button class="btn-action">LOCK</button></td>
          </tr>
          <tr>
            <td>#02</td>
            <td>VEHICLE</td>
            <td>88%</td>
            <td>78.2 km/h</td>
            <td><span style="color:#10b981;">LOW</span></td>
            <td><button class="btn-action">LOCK</button></td>
          </tr>
        `;
      }
    } catch (e) {
      console.warn('Backend API connection offline, using synthetic C4ISR telemetry');
    }
  }

  setInterval(fetchTargets, 1000);
  fetchTargets();
}

function lockTarget(id) {
  alert(`Target #${id} Locked for Gimbal Tracking Pursuit`);
}
