(() => {
  const canvas = document.getElementById('oscillator');
  if (!canvas) return;

  const ctx = canvas.getContext('2d', { alpha: false });
  const baseScale = 1000.0;
  const drops = [{ t: 5, x: 0.0, y: 0.0, amp: 150.0, sigma: 10.0 }];
  const vmax = 150.0 * 0.9;
  const speed = 0.75;
  let animationId = null;

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  const createSimulation = (viewportWidth, viewportHeight) => {
    const simMax = 300;
    let nx = simMax;
    let ny = Math.round(simMax * (viewportHeight / viewportWidth));
    if (viewportHeight > viewportWidth) {
      ny = simMax;
      nx = Math.round(simMax * (viewportWidth / viewportHeight));
    }

    nx = clamp(nx, 120, 420);
    ny = clamp(ny, 120, 420);

    const Lx = baseScale;
    const Ly = baseScale * (ny / nx);
    const dx = Lx / (nx - 1);
    const dy = Ly / (ny - 1);

    const X = new Float32Array(nx * ny);
    const Y = new Float32Array(nx * ny);
    for (let j = 0; j < ny; j += 1) {
      const y = -Ly / 2 + j * dy;
      for (let i = 0; i < nx; i += 1) {
        const x = -Lx / 2 + i * dx;
        const idx = j * nx + i;
        X[idx] = x;
        Y[idx] = y;
      }
    }

    const c = 400.0;
    const courant = 0.5;
    const dt = courant * Math.min(dx, dy) / c;
    const c2dt2 = (c * dt) * (c * dt);

    const dampGlobal = 0.0;
    const spongeWidth = Math.floor(Math.min(nx, ny) * 0.12);
    const sponge = new Float32Array(nx * ny);
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const idx = j * nx + i;
        const di = Math.min(j, ny - 1 - j);
        const dj = Math.min(i, nx - 1 - i);
        const dmin = Math.min(di, dj);
        if (dmin < spongeWidth) {
          const frac = (spongeWidth - dmin) / spongeWidth;
          sponge[idx] = 1.0 - 0.9 * (frac * frac);
        } else {
          sponge[idx] = 1.0;
        }
      }
    }

    let uPrev = new Float32Array(nx * ny);
    let u = new Float32Array(nx * ny);
    let uNext = new Float32Array(nx * ny);
    let frameIndex = 0;
    let speedAccumulator = 0;

    const dropsByStep = new Map();
    drops.forEach((drop) => {
      if (!dropsByStep.has(drop.t)) {
        dropsByStep.set(drop.t, []);
      }
      dropsByStep.get(drop.t).push(drop);
    });

    const addDrop = (field, xPos, yPos, amplitude, sigma) => {
      const sigma2 = 2.0 * sigma * sigma;
      for (let j = 0; j < ny; j += 1) {
        for (let i = 0; i < nx; i += 1) {
          const idx = j * nx + i;
          const dxp = X[idx] - xPos;
          const dyp = Y[idx] - yPos;
          const r2 = dxp * dxp + dyp * dyp;
          field[idx] += amplitude * Math.exp(-r2 / sigma2);
        }
      }
    };

    const image = ctx.createImageData(nx, ny);
    const pixels = image.data;
    const offscreen = document.createElement('canvas');
    offscreen.width = nx;
    offscreen.height = ny;
    const offscreenCtx = offscreen.getContext('2d');

    const render = () => {
      for (let j = 0; j < ny; j += 1) {
        for (let i = 0; i < nx; i += 1) {
          const idx = j * nx + i;
          const p = idx * 4;

          const value = u[idx];
          let t = (value / vmax + 1) / 2;
          if (t < 0) t = 0;
          if (t > 1) t = 1;

          let intensity;
          if (t < 0.5) {
            intensity = Math.round((t / 0.5) * 255);
          } else {
            intensity = Math.round((1 - (t - 0.5) / 0.5) * 255);
          }

          pixels[p] = intensity;
          pixels[p + 1] = intensity;
          pixels[p + 2] = intensity;
          pixels[p + 3] = 255;
        }
      }

      offscreenCtx.putImageData(image, 0, 0);
      ctx.clearRect(0, 0, viewportWidth, viewportHeight);
      ctx.drawImage(offscreen, 0, 0, viewportWidth, viewportHeight);
    };

    const updateSimulation = () => {
      const dropsNow = dropsByStep.get(frameIndex);
      if (dropsNow) {
        dropsNow.forEach((drop) => {
          addDrop(u, drop.x, drop.y, drop.amp, drop.sigma);
        });
      }

      for (let j = 1; j < ny - 1; j += 1) {
        for (let i = 1; i < nx - 1; i += 1) {
          const idx = j * nx + i;
          const lap = (u[idx - 1] + u[idx + 1] + u[idx - nx] + u[idx + nx] - 4.0 * u[idx]) / (dx * dy);
          let next = 2.0 * u[idx] - uPrev[idx] + c2dt2 * lap;
          next *= (1.0 - dampGlobal);
          next *= sponge[idx];
          uNext[idx] = next;
        }
      }

      for (let i = 0; i < nx; i += 1) {
        uNext[i] = 0.0;
        uNext[(ny - 1) * nx + i] = 0.0;
      }
      for (let j = 0; j < ny; j += 1) {
        uNext[j * nx] = 0.0;
        uNext[j * nx + (nx - 1)] = 0.0;
      }

      const temp = uPrev;
      uPrev = u;
      u = uNext;
      uNext = temp;

      frameIndex += 1;
    };

    const step = () => {
      speedAccumulator += speed;
      while (speedAccumulator >= 1) {
        updateSimulation();
        speedAccumulator -= 1;
      }

      render();
      animationId = requestAnimationFrame(step);
    };

    return { step, render };
  };

  const resize = () => {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }

    const dpr = window.devicePixelRatio || 1;
    const viewportWidth = Math.max(1, Math.floor(window.innerWidth));
    const viewportHeight = Math.max(1, Math.floor(window.innerHeight));
    canvas.width = Math.floor(viewportWidth * dpr);
    canvas.height = Math.floor(viewportHeight * dpr);
    canvas.style.width = `${viewportWidth}px`;
    canvas.style.height = `${viewportHeight}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const simulation = createSimulation(viewportWidth, viewportHeight);
    simulation.render();
    animationId = requestAnimationFrame(simulation.step);
  };

  window.addEventListener('resize', resize);
  resize();
})();
