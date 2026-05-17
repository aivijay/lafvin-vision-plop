// LAFVIN Vision PLOP — Laptop Frontend
// Receives SSE: robot status + depth (from laptop) + depth analysis
// Camera shown via <img src="/camera/live"> direct MJPEG from RPi (not relayed through SSE)

const SERVER = window.location.origin;

// ─── State ───────────────────────────────────────────────────────────────────
let evtSource = null;
let connected = false;
let frameCount = 0, depthCount = 0;
let fpsInterval = null;
let lastDepthResult = null;

// ─── Depth canvas — updated on demand via SSE depth events ──────────────────
// No polling — depth image only refreshed when SSE delivers a new depth result
// Saves 120 HTTP requests/minute from eliminated 0.5s polling interval

function initDepthCanvas() {
  const canvas = document.getElementById('depth-canvas');
  if (!canvas) return;
  // No polling — updateDepthCanvasFromSSE() handles it when depth arrives via SSE
}

// Called by handleUpdate() when SSE brings a fresh depth result
function updateDepthCanvasFromSSE(depth) {
  const canvas = document.getElementById('depth-canvas');
  if (canvas && depth) {
    // Cache-bust to force browser reload of the new cached JPEG
    const ts = Date.now();
    canvas.src = `/depth/colorized.jpg?t=${ts}`;
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  connectSSE();
  startFPSCounter();
  initDepthCanvas();
  document.getElementById('rpi-url').textContent = `RPi: ${SERVER}`;
});

// ─── SSE ─────────────────────────────────────────────────────────────────────
function connectSSE() {
  evtSource = new EventSource(`${SERVER}/ws/updates`);

  evtSource.onopen = () => {
    connected = true;
    setStatus('connected');
    console.log('[SSE] Connected');
  };

  evtSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('[SSE] received event, has depth:', !!data.depth, 'has robot:', !!data.robot);
    handleUpdate(data);
    frameCount++;
    if (data.depth) depthCount++;
    document.getElementById('last-update').textContent =
      `Last: ${new Date().toLocaleTimeString()}`;
  };

  evtSource.onerror = (err) => {
    connected = false;
    setStatus('error');
    console.error('[SSE] error:', err);
    evtSource.close();
    setTimeout(connectSSE, 500);
  };
}

function setStatus(state) {
  const el = document.getElementById('status');
  if (state === 'connected') {
    el.textContent = 'Connected'; el.className = 'badge connected';
  } else if (state === 'error') {
    el.textContent = 'Disconnected'; el.className = 'badge error';
  } else {
    el.textContent = 'Connecting...'; el.className = 'badge connecting';
  }
}

function startFPSCounter() {
  if (fpsInterval) clearInterval(fpsInterval);
  fpsInterval = setInterval(() => {
    document.getElementById('fps-display').textContent = `${frameCount} fps`;
    document.getElementById('depth-fps').textContent = `depth: ${depthCount}/s`;
    frameCount = 0;
    depthCount = 0;
  }, 1000);
}

// ─── Handle SSE update ────────────────────────────────────────────────────────
function handleUpdate(data) {
  // Robot status (from RPi)
  if (data.robot) {
    document.getElementById('ultrasonic').textContent =
      data.robot.ultrasonic_cm != null ? `${data.robot.ultrasonic_cm} cm` : '-- cm';
    document.getElementById('motors').textContent =
      data.robot.motors_on ? '🟢 ON' : '🔴 OFF';
    document.getElementById('gimbal-h').textContent =
      data.robot.gimbal_h != null ? `${data.robot.gimbal_h}°` : '--°';
    document.getElementById('gimbal-v').textContent =
      data.robot.gimbal_v != null ? `${data.robot.gimbal_v}°` : '--°';
  }

  // Depth + nav (from laptop Depth Anything V2)
  if (data.depth) {
    lastDepthResult = data.depth;
    updateDepthDisplay(data.depth);
    updateNavDisplay(data.depth);
    updateDepthCanvasFromSSE(data.depth);
    document.getElementById('model-status').textContent = 'Ready (vits)';
  }
}

// ─── Depth display ────────────────────────────────────────────────────────────
function updateDepthDisplay(depth) {
  document.getElementById('depth-info').textContent =
    `center: ${depth.depth_m}m · mean: ${depth.mean_depth_m}m · ` +
    `clear: ${depth.clear_path} · action: ${depth.suggested_action}`;

  document.getElementById('center-depth').textContent =
    depth.depth_m != null ? `${depth.depth_m}m` : '--';
  document.getElementById('obstacle-pct').textContent =
    depth.obstacle_pct != null ? `${Math.round(depth.obstacle_pct * 100)}%` : '--';
  document.getElementById('near-pct').textContent =
    depth.near_pct != null ? `${Math.round(depth.near_pct * 100)}%` : '--';
}

// ─── Nav display ──────────────────────────────────────────────────────────────
function updateNavDisplay(depth) {
  const navStatus = document.getElementById('nav-status');
  const actionBox = document.getElementById('suggested-action');
  const detail = document.getElementById('obstacle-detail');

  if (depth.clear_path) {
    navStatus.textContent = '✅ CLEAR PATH';
    navStatus.className = 'status-large clear';
  } else if (depth.obstacle_detected) {
    navStatus.textContent = '⚠️ OBSTACLE';
    navStatus.className = 'status-large obstacle';
  } else if (depth.suggested_action === 'stop') {
    navStatus.textContent = '🛑 STOP';
    navStatus.className = 'status-large warning';
  } else {
    navStatus.textContent = '⚠️ CHECK PATH';
    navStatus.className = 'status-large warning';
  }

  actionBox.textContent = `Action: ${depth.suggested_action || '--'}`;

  const dist = depth.distance_to_obstacle_cm;
  if (depth.clear_path) {
    detail.textContent = `Path clear beyond ${dist}cm`;
  } else {
    detail.textContent = `Obstacle at ${dist}cm (${Math.round(depth.obstacle_pct * 100)}% floor)`;
  }

  // Update robot panel clear/obstacle flags
  document.getElementById('clear-path').textContent = depth.clear_path ? '✅ Yes' : '❌ No';
  document.getElementById('obstacle-flag').textContent = depth.obstacle_detected ? '⚠️ Yes' : '✅ No';

  // Calibration info
  if (depth.calibration) {
    const c = depth.calibration;
    document.getElementById('calib-params').textContent =
      `4z [${c.thresholds}] [${c.scales.join(',')}]`;
  }
}

// ─── Robot commands ────────────────────────────────────────────────────────────
async function sendCommand(action, speed) {
  try {
    const resp = await fetch(`${SERVER}/robot/command`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, speed }),
    });
    const result = await resp.json();
    console.log('[command]', action, result);
    return result;
  } catch (e) {
    console.error('[command] failed:', e);
    return { ok: false };
  }
}