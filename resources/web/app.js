(function () {
  'use strict';

  const statusEl = document.getElementById('status-indicator');
  const frameInfoEl = document.getElementById('frame-info');
  const imageEl = document.getElementById('current-image');
  const confSlider = document.getElementById('conf-threshold');
  const confValueEl = document.getElementById('conf-value');
  const canvas = document.getElementById('viewer');

  const gl = canvas.getContext('webgl2', { antialias: true });
  if (!gl) {
    statusEl.textContent = 'WebGL2 not supported';
    statusEl.style.color = '#ff6b6b';
    return;
  }

  // ---------------------------------------------------------------------------
  // Matrix utilities (minimal replacements for a subset of glMatrix)
  // ---------------------------------------------------------------------------
  function mat4Perspective(out, fovy, aspect, near, far) {
    const f = 1.0 / Math.tan(fovy / 2);
    out[0] = f / aspect;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;

    out[4] = 0;
    out[5] = f;
    out[6] = 0;
    out[7] = 0;

    out[8] = 0;
    out[9] = 0;
    out[10] = (far + near) / (near - far);
    out[11] = -1;

    out[12] = 0;
    out[13] = 0;
    out[14] = (2 * far * near) / (near - far);
    out[15] = 0;
    return out;
  }

  function mat4LookAt(out, eye, center, up) {
    let x0, x1, x2;
    let y0, y1, y2;
    let z0, z1, z2;
    let len;

    z0 = eye[0] - center[0];
    z1 = eye[1] - center[1];
    z2 = eye[2] - center[2];
    len = Math.hypot(z0, z1, z2);
    if (len === 0) {
      z2 = 1;
      len = 1;
    }
    z0 /= len;
    z1 /= len;
    z2 /= len;

    x0 = up[1] * z2 - up[2] * z1;
    x1 = up[2] * z0 - up[0] * z2;
    x2 = up[0] * z1 - up[1] * z0;
    len = Math.hypot(x0, x1, x2);
    if (len === 0) {
      x0 = 0;
      x1 = 0;
      x2 = 0;
    } else {
      x0 /= len;
      x1 /= len;
      x2 /= len;
    }

    y0 = z1 * x2 - z2 * x1;
    y1 = z2 * x0 - z0 * x2;
    y2 = z0 * x1 - z1 * x0;

    out[0] = x0;
    out[1] = y0;
    out[2] = z0;
    out[3] = 0;

    out[4] = x1;
    out[5] = y1;
    out[6] = z1;
    out[7] = 0;

    out[8] = x2;
    out[9] = y2;
    out[10] = z2;
    out[11] = 0;

    out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
    out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
    out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
    out[15] = 1;

    return out;
  }

  function mat3FromMat4(out, m) {
    out[0] = m[0];
    out[1] = m[1];
    out[2] = m[2];
    out[3] = m[4];
    out[4] = m[5];
    out[5] = m[6];
    out[6] = m[8];
    out[7] = m[9];
    out[8] = m[10];
    return out;
  }

  function mat3Identity(out) {
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 1;
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
    out[8] = 1;
    return out;
  }

  const BASIS_SIGNS = [1, -1, -1, 1];

  function cvToGlVec3(vec) {
    if (!Array.isArray(vec) || vec.length < 3) {
      return vec;
    }
    return [vec[0], -vec[1], -vec[2]];
  }

  function cvToGlBuffer(buffer) {
    for (let i = 0; i + 2 < buffer.length; i += 3) {
      buffer[i + 1] = -buffer[i + 1];
      buffer[i + 2] = -buffer[i + 2];
    }
    return buffer;
  }

  function cvToGlPose(pose) {
    if (!Array.isArray(pose) || pose.length === 0) {
      return pose;
    }
    const rows = Math.min(4, pose.length);
    const cols = Math.min(4, Array.isArray(pose[0]) ? pose[0].length : 0);
    if (cols === 0) {
      return pose;
    }
    const result = new Array(4);
    for (let i = 0; i < 4; i += 1) {
      result[i] = new Array(4);
      for (let j = 0; j < 4; j += 1) {
        if (i < rows && j < cols) {
          const baseRow = pose[i];
          const value = Array.isArray(baseRow) ? baseRow[j] : undefined;
          if (typeof value === 'number') {
            const signI = BASIS_SIGNS[i] ?? 1;
            const signJ = BASIS_SIGNS[j] ?? 1;
            result[i][j] = value * signI * signJ;
          } else {
            result[i][j] = j === i ? 1 : 0;
          }
        } else if (i === j) {
          result[i][j] = 1;
        } else {
          result[i][j] = 0;
        }
      }
    }
    return result;
  }

  function mat3InvertTranspose(out, m) {
    const a00 = m[0];
    const a01 = m[1];
    const a02 = m[2];
    const a10 = m[3];
    const a11 = m[4];
    const a12 = m[5];
    const a20 = m[6];
    const a21 = m[7];
    const a22 = m[8];

    const b01 = a22 * a11 - a12 * a21;
    const b11 = -a22 * a10 + a12 * a20;
    const b21 = a21 * a10 - a11 * a20;

    let det = a00 * b01 + a01 * b11 + a02 * b21;
    if (!det) {
      return mat3Identity(out);
    }
    det = 1.0 / det;

    out[0] = b01 * det;
    out[1] = (-a22 * a01 + a02 * a21) * det;
    out[2] = (a12 * a01 - a02 * a11) * det;
    out[3] = b11 * det;
    out[4] = (a22 * a00 - a02 * a20) * det;
    out[5] = (-a12 * a00 + a02 * a10) * det;
    out[6] = b21 * det;
    out[7] = (-a21 * a00 + a01 * a20) * det;
    out[8] = (a11 * a00 - a01 * a10) * det;

    return out;
  }

  // ---------------------------------------------------------------------------
  // Shader compilation helpers
  // ---------------------------------------------------------------------------
  function compileShader(glContext, type, source) {
    const shader = glContext.createShader(type);
    glContext.shaderSource(shader, source);
    glContext.compileShader(shader);
    if (!glContext.getShaderParameter(shader, glContext.COMPILE_STATUS)) {
      const info = glContext.getShaderInfoLog(shader);
      glContext.deleteShader(shader);
      throw new Error(`Shader compile error: ${info}`);
    }
    return shader;
  }

  function createProgram(glContext, vertexSource, fragmentSource) {
    const program = glContext.createProgram();
    const vs = compileShader(glContext, glContext.VERTEX_SHADER, vertexSource);
    const fs = compileShader(glContext, glContext.FRAGMENT_SHADER, fragmentSource);
    glContext.attachShader(program, vs);
    glContext.attachShader(program, fs);
    glContext.linkProgram(program);
    if (!glContext.getProgramParameter(program, glContext.LINK_STATUS)) {
      const info = glContext.getProgramInfoLog(program);
      glContext.deleteProgram(program);
      throw new Error(`Program link error: ${info}`);
    }
    return program;
  }

  // ---------------------------------------------------------------------------
  // Shader sources (aligned with surfelmap.glsl lighting model)
  // ---------------------------------------------------------------------------
  const pointVertexShader = `#version 300 es\n\n  precision highp float;\n\n  layout(location = 0) in vec3 position;\n  layout(location = 1) in vec3 normal;\n  layout(location = 2) in vec3 color;\n\n  uniform mat4 modelViewMatrix;\n  uniform mat4 projectionMatrix;\n  uniform mat3 normalMatrix;\n  uniform float pointSize;\n\n  out vec3 vNormal;\n  out vec3 vColor;\n  out float vDepth;\n\n  void main() {\n    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);\n    vDepth = -mvPosition.z;\n    gl_Position = projectionMatrix * mvPosition;\n    gl_PointSize = clamp(pointSize / max(0.05, vDepth), 2.0, 24.0);\n    vNormal = normalize(normalMatrix * normal);\n    vColor = color;\n  }\n`;

  const pointFragmentShader = `#version 300 es\n\n  precision highp float;\n\n  in vec3 vNormal;\n  in vec3 vColor;\n\n  uniform vec3 lightDirection;\n  uniform vec3 phong;\n  uniform bool showNormal;\n\n  out vec4 fragColor;\n\n  void main() {\n    vec2 uv = gl_PointCoord * 2.0 - 1.0;\n    float dist = dot(uv, uv);\n    if (dist > 1.0) {\n      discard;\n    }\n\n    vec3 N = normalize(vNormal);\n    if (showNormal) {\n      vec3 shaded = clamp((vec3(N.x, -N.y, -N.z) * 0.5) + 0.5, 0.0, 1.0);\n      fragColor = vec4(shaded, 1.0);\n      return;\n    }\n\n    vec3 L = normalize(lightDirection);\n    float lambert = max(dot(N, L), 0.0);\n    vec3 V = vec3(0.0, 0.0, 1.0);\n    vec3 R = reflect(-L, N);\n    float spec = pow(max(dot(R, V), 0.0), phong.z);\n\n    vec3 color = vColor * (phong.x + lambert * phong.y) + spec * 0.8;\n    fragColor = vec4(color, 1.0);\n  }\n`;

  const lineVertexShader = `#version 300 es\n\n  precision highp float;\n\n  layout(location = 0) in vec3 position;\n\n  uniform mat4 modelViewMatrix;\n  uniform mat4 projectionMatrix;\n\n  void main() {\n    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n  }\n`;

  const lineFragmentShader = `#version 300 es\n\n  precision highp float;\n\n  uniform vec3 color;\n  out vec4 fragColor;\n\n  void main() {\n    fragColor = vec4(color, 1.0);\n  }\n`;

  const pointProgram = createProgram(gl, pointVertexShader, pointFragmentShader);
  const lineProgram = createProgram(gl, lineVertexShader, lineFragmentShader);

  const pointUniforms = {
    modelViewMatrix: gl.getUniformLocation(pointProgram, 'modelViewMatrix'),
    projectionMatrix: gl.getUniformLocation(pointProgram, 'projectionMatrix'),
    normalMatrix: gl.getUniformLocation(pointProgram, 'normalMatrix'),
    pointSize: gl.getUniformLocation(pointProgram, 'pointSize'),
    lightDirection: gl.getUniformLocation(pointProgram, 'lightDirection'),
    phong: gl.getUniformLocation(pointProgram, 'phong'),
    showNormal: gl.getUniformLocation(pointProgram, 'showNormal'),
  };

  const lineUniforms = {
    modelViewMatrix: gl.getUniformLocation(lineProgram, 'modelViewMatrix'),
    projectionMatrix: gl.getUniformLocation(lineProgram, 'projectionMatrix'),
    color: gl.getUniformLocation(lineProgram, 'color'),
  };

  function createPointVAO() {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const normalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    return {
      vao,
      positionBuffer,
      normalBuffer,
      colorBuffer,
    };
  }

  function createLineVAO() {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return { vao, buffer };
  }

  const surfelVAO = createPointVAO();
  const keyframeVAO = createPointVAO();
  const trajectoryVAO = createLineVAO();
  const edgeVAO = createLineVAO();
  const currentAxisVAOs = {
    x: createLineVAO(),
    y: createLineVAO(),
    z: createLineVAO(),
  };
  const keyframeAxisVAOs = {
    x: createLineVAO(),
    y: createLineVAO(),
    z: createLineVAO(),
  };

  let surfelCount = 0;
  let keyframeCount = 0;
  let trajectoryVertexCount = 0;
  let edgeVertexCount = 0;
  const currentAxisCounts = { x: 0, y: 0, z: 0 };
  const keyframeAxisCounts = { x: 0, y: 0, z: 0 };

  // ---------------------------------------------------------------------------
  // Camera state and controls
  // ---------------------------------------------------------------------------
  const camera = {
    yaw: Math.PI * 0.25,
    pitch: Math.PI * 0.20,
    distance: 3.5,
    minDistance: 0.35,
    maxDistance: 80.0,
  };

  const target = new Float32Array([0, 0, 0]);
  const desiredTarget = new Float32Array([0, 0, 0]);

  let pointerActive = false;
  let lastPointerX = 0;
  let lastPointerY = 0;

  canvas.addEventListener('pointerdown', (event) => {
    pointerActive = true;
    lastPointerX = event.clientX;
    lastPointerY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener('pointermove', (event) => {
    if (!pointerActive) {
      return;
    }
    const dx = event.clientX - lastPointerX;
    const dy = event.clientY - lastPointerY;
    lastPointerX = event.clientX;
    lastPointerY = event.clientY;

    camera.yaw -= dx * 0.005;
    camera.pitch -= dy * 0.005;
    const limit = Math.PI * 0.49;
    camera.pitch = Math.max(-limit, Math.min(limit, camera.pitch));
  });

  function endPointer() {
    pointerActive = false;
  }
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointerleave', endPointer);
  canvas.addEventListener('pointercancel', endPointer);

  canvas.addEventListener(
    'wheel',
    (event) => {
      event.preventDefault();
      const factor = Math.exp(event.deltaY * 0.0015);
      camera.distance *= factor;
      camera.distance = Math.max(
        camera.minDistance,
        Math.min(camera.maxDistance, camera.distance),
      );
    },
    { passive: false },
  );

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    return { width, height };
  }

  // ---------------------------------------------------------------------------
  // Scene data upload helpers
  // ---------------------------------------------------------------------------
  function updateCurrentAxes(cameraPose) {
    currentAxisCounts.x = 0;
    currentAxisCounts.y = 0;
    currentAxisCounts.z = 0;
    const axisKeys = ['x', 'y', 'z'];
    const empty = new Float32Array(0);
    const bindEmpty = () => {
      axisKeys.forEach((key) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, currentAxisVAOs[key].buffer);
        gl.bufferData(gl.ARRAY_BUFFER, empty, gl.DYNAMIC_DRAW);
      });
    };

    if (!cameraPose || !Array.isArray(cameraPose) || cameraPose.length < 3) {
      bindEmpty();
      return;
    }
    const pose = cvToGlPose(cameraPose);
    if (!Array.isArray(pose) || pose.length < 3 || !Array.isArray(pose[0]) || pose[0].length < 3) {
      bindEmpty();
      return;
    }

    const origin = [pose[0][3] || 0, pose[1][3] || 0, pose[2][3] || 0];
    const axes = [
      [pose[0][0], pose[1][0], pose[2][0]],
      [pose[0][1], pose[1][1], pose[2][1]],
      [pose[0][2], pose[1][2], pose[2][2]],
    ];
    const axisLength = 0.2;

    axisKeys.forEach((key, idx) => {
      const axis = axes[idx];
      if (!axis) {
        gl.bindBuffer(gl.ARRAY_BUFFER, currentAxisVAOs[key].buffer);
        gl.bufferData(gl.ARRAY_BUFFER, empty, gl.DYNAMIC_DRAW);
        return;
      }
      const end = [
        origin[0] + axis[0] * axisLength,
        origin[1] + axis[1] * axisLength,
        origin[2] + axis[2] * axisLength,
      ];
      const data = new Float32Array([
        origin[0], origin[1], origin[2],
        end[0], end[1], end[2],
      ]);
      gl.bindBuffer(gl.ARRAY_BUFFER, currentAxisVAOs[key].buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      currentAxisCounts[key] = data.length / 3;
    });
  }


  function updateSurfels(scene) {
    if (!scene || !Array.isArray(scene.points) || scene.points.length === 0) {
      surfelCount = 0;
      updateCurrentAxes(scene ? scene.cameraPose : null);
      return;
    }
    const positions = cvToGlBuffer(new Float32Array(scene.points));
    const normals = cvToGlBuffer(new Float32Array(scene.normals));
    const colors = new Float32Array(scene.colors);
    surfelCount = positions.length / 3;

    gl.bindBuffer(gl.ARRAY_BUFFER, surfelVAO.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, surfelVAO.normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, surfelVAO.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);

    updateCurrentAxes(scene.cameraPose);
  }

  function updateKeyframes(entries) {
    const axisKeys = ['x', 'y', 'z'];
    const empty = new Float32Array(0);
    if (!Array.isArray(entries) || entries.length === 0) {
      keyframeCount = 0;
      trajectoryVertexCount = 0;
      edgeVertexCount = 0;
      axisKeys.forEach((key) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, keyframeAxisVAOs[key].buffer);
        gl.bufferData(gl.ARRAY_BUFFER, empty, gl.DYNAMIC_DRAW);
        keyframeAxisCounts[key] = 0;
      });
      return;
    }

    const count = entries.length;
    const positions = new Float32Array(count * 3);
    const normals = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const axisData = { x: [], y: [], z: [] };
    const axisLength = 0.12;

    for (let i = 0; i < count; i += 1) {
      const base = i * 3;
      const pos = cvToGlVec3(entries[i].position || [0, 0, 0]);
      positions[base + 0] = pos[0];
      positions[base + 1] = pos[1];
      positions[base + 2] = pos[2];
      const normal = [0, 1, 0];
      normals[base + 0] = normal[0];
      normals[base + 1] = normal[1];
      normals[base + 2] = normal[2];
      if (i === count - 1) {
        colors[base + 0] = 1.0;
        colors[base + 1] = 0.4;
        colors[base + 2] = 0.4;
      } else {
        colors[base + 0] = 0.1;
        colors[base + 1] = 0.8;
        colors[base + 2] = 1.0;
      }

      const pose = cvToGlPose(entries[i].pose);
      if (pose && Array.isArray(pose) && pose.length >= 3 && Array.isArray(pose[0]) && pose[0].length >= 3) {
        const origin = [pose[0][3] || 0, pose[1][3] || 0, pose[2][3] || 0];
        const axes = [
          [pose[0][0], pose[1][0], pose[2][0]],
          [pose[0][1], pose[1][1], pose[2][1]],
          [pose[0][2], pose[1][2], pose[2][2]],
        ];
        axisKeys.forEach((key, idx) => {
          const axis = axes[idx];
          if (!axis) {
            return;
          }
          const end = [
            origin[0] + axis[0] * axisLength,
            origin[1] + axis[1] * axisLength,
            origin[2] + axis[2] * axisLength,
          ];
          axisData[key].push(
            origin[0], origin[1], origin[2],
            end[0], end[1], end[2],
          );
        });
      }
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, keyframeVAO.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, keyframeVAO.normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, keyframeVAO.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
    keyframeCount = count;

    axisKeys.forEach((key) => {
      const data = new Float32Array(axisData[key]);
      gl.bindBuffer(gl.ARRAY_BUFFER, keyframeAxisVAOs[key].buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      keyframeAxisCounts[key] = data.length / 3;
    });

    if (count >= 2) {
      const trajPositions = new Float32Array((count - 1) * 6);
      for (let i = 0; i < count - 1; i += 1) {
        const a = cvToGlVec3(entries[i].position || [0, 0, 0]);
        const b = cvToGlVec3(entries[i + 1].position || [0, 0, 0]);
        const base = i * 6;
        trajPositions.set(a, base);
        trajPositions.set(b, base + 3);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, trajectoryVAO.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, trajPositions, gl.DYNAMIC_DRAW);
      trajectoryVertexCount = trajPositions.length / 3;
    } else {
      trajectoryVertexCount = 0;
    }
  }

  function updateEdges(edges) {
    if (!Array.isArray(edges) || edges.length === 0) {
      edgeVertexCount = 0;
      return;
    }
    const positions = new Float32Array(edges.length * 6);
    for (let i = 0; i < edges.length; i += 1) {
      const base = i * 6;
      const segment = edges[i].points || [[0, 0, 0], [0, 0, 0]];
      const a = cvToGlVec3(segment[0] || [0, 0, 0]);
      const b = cvToGlVec3(segment[1] || [0, 0, 0]);
      positions.set(a, base);
      positions.set(b, base + 3);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, edgeVAO.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    edgeVertexCount = positions.length / 3;
  }

  // ---------------------------------------------------------------------------
  // Fetch + UI wiring
  // ---------------------------------------------------------------------------
  let lastFetchTime = 0;

  confSlider.addEventListener('input', (event) => {
    const value = Number.parseFloat(event.target.value);
    confValueEl.textContent = value.toFixed(2);
  });

  confSlider.addEventListener('change', async (event) => {
    const value = Number.parseFloat(event.target.value);
    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confThreshold: value }),
      });
    } catch (error) {
      statusEl.textContent = 'Config update failed';
      statusEl.style.color = '#ff6b6b';
    }
  });

  async function fetchState() {
    try {
      const response = await fetch('/api/state', { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      applyState(data);
      statusEl.textContent = data.ready ? 'Streaming' : 'Waiting for surfels…';
      statusEl.style.color = data.ready ? '#66ff88' : '#f0b429';
      lastFetchTime = performance.now();
    } catch (error) {
      statusEl.textContent = 'Connection lost';
      statusEl.style.color = '#ff6b6b';
    }
  }

  function applyState(state) {
    if (state.currentFrame) {
      frameInfoEl.textContent = `Frame ${state.currentFrame.id} • Surfel points: ${state.currentFrame.pointCount}`;
      imageEl.src = state.currentFrame.image;
      const posCv = state.currentFrame.position || [0, 0, 0];
      const pos = cvToGlVec3(posCv);
      desiredTarget[0] = pos[0];
      desiredTarget[1] = pos[1];
      desiredTarget[2] = pos[2];
    }

    updateSurfels(state.scene);
    updateKeyframes(state.keyframes);
    updateEdges(state.edges);

    if (typeof state.confThreshold === 'number' && !confSlider.matches(':active')) {
      confSlider.value = state.confThreshold.toFixed(1);
      confValueEl.textContent = state.confThreshold.toFixed(2);
    }
  }

  function smoothTargets(alpha) {
    for (let i = 0; i < 3; i += 1) {
      target[i] += (desiredTarget[i] - target[i]) * alpha;
    }
  }

  const viewMatrix = new Float32Array(16);
  const projectionMatrix = new Float32Array(16);
  const normalMatrix = new Float32Array(9);

  function render() {
    const { width, height } = resizeCanvas();
    smoothTargets(0.1);

    const eye = [
      target[0] + camera.distance * Math.cos(camera.pitch) * Math.sin(camera.yaw),
      target[1] + camera.distance * Math.sin(camera.pitch),
      target[2] + camera.distance * Math.cos(camera.pitch) * Math.cos(camera.yaw),
    ];

    mat4LookAt(viewMatrix, eye, target, [0, 1, 0]);
    mat4Perspective(projectionMatrix, Math.PI / 3, width / height, 0.05, 100.0);
    mat3FromMat4(normalMatrix, viewMatrix);
    mat3InvertTranspose(normalMatrix, normalMatrix);

    gl.clearColor(0.06, 0.07, 0.08, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);

    gl.useProgram(pointProgram);
    gl.uniformMatrix4fv(pointUniforms.modelViewMatrix, false, viewMatrix);
    gl.uniformMatrix4fv(pointUniforms.projectionMatrix, false, projectionMatrix);
    gl.uniformMatrix3fv(pointUniforms.normalMatrix, false, normalMatrix);
    gl.uniform3f(pointUniforms.lightDirection, 0.4, 0.8, 0.2);
    gl.uniform3f(pointUniforms.phong, 0.3, 0.7, 24.0);
    gl.uniform1i(pointUniforms.showNormal, 0);

    if (surfelCount > 0) {
      gl.uniform1f(pointUniforms.pointSize, 14.0);
      gl.bindVertexArray(surfelVAO.vao);
      gl.drawArrays(gl.POINTS, 0, surfelCount);
    }

    if (keyframeCount > 0) {
      gl.uniform1f(pointUniforms.pointSize, 10.0);
      gl.bindVertexArray(keyframeVAO.vao);
      gl.drawArrays(gl.POINTS, 0, keyframeCount);
    }

    gl.useProgram(lineProgram);
    gl.uniformMatrix4fv(lineUniforms.modelViewMatrix, false, viewMatrix);
    gl.uniformMatrix4fv(lineUniforms.projectionMatrix, false, projectionMatrix);

    if (trajectoryVertexCount > 0) {
      gl.uniform3f(lineUniforms.color, 1.0, 1.0, 1.0);
      gl.bindVertexArray(trajectoryVAO.vao);
      gl.drawArrays(gl.LINES, 0, trajectoryVertexCount);
    }

    if (edgeVertexCount > 0) {
      gl.uniform3f(lineUniforms.color, 0.4, 1.0, 0.6);
      gl.bindVertexArray(edgeVAO.vao);
      gl.drawArrays(gl.LINES, 0, edgeVertexCount);
    }

    const axisOrder = ['x', 'y', 'z'];
    const currentAxisColors = {
      x: [1.0, 0.3, 0.3],
      y: [0.3, 1.0, 0.3],
      z: [0.3, 0.6, 1.0],
    };
    const keyframeAxisColors = {
      x: [0.8, 0.35, 0.35],
      y: [0.35, 0.8, 0.35],
      z: [0.35, 0.55, 0.95],
    };

    axisOrder.forEach((key) => {
      if (currentAxisCounts[key] > 0) {
        const color = currentAxisColors[key];
        gl.uniform3f(lineUniforms.color, color[0], color[1], color[2]);
        gl.bindVertexArray(currentAxisVAOs[key].vao);
        gl.drawArrays(gl.LINES, 0, currentAxisCounts[key]);
      }
    });

    axisOrder.forEach((key) => {
      if (keyframeAxisCounts[key] > 0) {
        const color = keyframeAxisColors[key];
        gl.uniform3f(lineUniforms.color, color[0], color[1], color[2]);
        gl.bindVertexArray(keyframeAxisVAOs[key].vao);
        gl.drawArrays(gl.LINES, 0, keyframeAxisCounts[key]);
      }
    });

    if (performance.now() - lastFetchTime > 5000) {
      statusEl.textContent = 'Waiting for data…';
      statusEl.style.color = '#f0b429';
    }

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
  fetchState();
  setInterval(fetchState, 100);
})();
