(() => {
  const timeEl = document.getElementById('current-time');
  const dateEl = document.getElementById('current-date');

  const updateTime = () => {
    const now = new Date();

    if (timeEl) {
      const timeStr = now.toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
      timeEl.textContent = ` ${timeStr}`;
    }

    if (dateEl) {
      const dateStr = now.toLocaleDateString(undefined, {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
      });
      dateEl.textContent = ` ${dateStr}`;
    }
  };

  updateTime();
  setInterval(updateTime, 1000);
})();
