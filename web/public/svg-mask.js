function loadSVGMask() {
  const iconImg = document.getElementById('app-icon');
  if (!iconImg || iconImg.tagName !== 'IMG') return;

  const oscillatorCanvas = document.getElementById('oscillator');
  const layerTop = document.querySelector('.layer-top');

  fetch(iconImg.src)
    .then(res => res.text())
    .then(svgText => {
      const parser = new DOMParser();
      const svgDoc = parser.parseFromString(svgText, 'image/svg+xml');
      const svgElement = svgDoc.documentElement;

      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = oscillatorCanvas.width;
      maskCanvas.height = oscillatorCanvas.height;
      maskCanvas.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        mix-blend-mode: lighten;
      `;

      const maskCtx = maskCanvas.getContext('2d', { alpha: true });

      function updateMask() {
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

        try {
          const imageData = oscillatorCanvas.getContext('2d').getImageData(
            0, 0,
            oscillatorCanvas.width,
            oscillatorCanvas.height
          );
          const data = imageData.data;

          for (let i = 0; i < data.length; i += 4) {
            data[i] = 255 - data[i];
            data[i + 1] = 255 - data[i + 1];
            data[i + 2] = 255 - data[i + 2];
          }

          maskCtx.putImageData(imageData, 0, 0);
        } catch (e) {
          console.warn('CORS or canvas access issue:', e);
        }

        requestAnimationFrame(updateMask);
      }

      layerTop.innerHTML = '';
      layerTop.appendChild(maskCanvas);

      setTimeout(updateMask, 100);
      console.log('SVG mask with true pixel inversion loaded');
    })
    .catch(err => console.error('Failed to load SVG mask:', err));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', loadSVGMask);
} else {
  loadSVGMask();
}

export { loadSVGMask };
