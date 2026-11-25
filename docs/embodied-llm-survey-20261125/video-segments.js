// Each chapter is a time range in the original, uncut experiment recording.
function setupVideoSegments(video, buttons, fullButton, speed, status, rounds, clock) {
  let selection = null;
  let ownSeek = false;
  let version = 0;
  let frame = null;
  const rangeText = s => `${s.start.toFixed(2)}–${s.end.toFixed(2)}秒`;
  function highlight(button) {
    buttons.forEach(b => b.setAttribute('aria-pressed', String(Boolean(button) && b.dataset.start === button.dataset.start && b.dataset.end === button.dataset.end)));
  }
  function describe() {
    if (selection) {
      const state = selection.complete ? '本轮播放结束，已暂停' : video.paused ? '已暂停' : '正在播放';
      status.textContent = `${state}：${selection.label} · ${rangeText(selection)}（原始完整视频中的动作段）`;
    } else {
      const row = rounds.find(r => video.currentTime >= r.start && video.currentTime < r.end);
      const current = row ? `第${row.call}轮 · ${rangeText(row)}` : '初始／结束画面';
      status.textContent = `完整视频模式 · 当前${current}；可选下方关键轮次，到该轮终点自动暂停。`;
    }
  }
  function seek(time) {
    if (Math.abs(video.currentTime - time) <= 0.000001) return;
    ownSeek = true;
    video.currentTime = time;
  }
  function stopFrame() {
    if (frame !== null) clock.cancelAnimationFrame(frame);
    frame = null;
  }
  function checkEnd() {
    if (selection && !selection.complete && !video.seeking && video.currentTime >= selection.end) {
      selection.complete = true;
      video.pause();
      seek(selection.end); // Keep the action's final frame visible.
    }
    describe();
  }
  function tick() {
    frame = null;
    checkEnd();
    if (!video.paused && selection && !selection.complete) frame = clock.requestAnimationFrame(tick);
  }
  function playRange(next, button) {
    const request = ++version;
    video.pause();
    stopFrame();
    selection = next;
    highlight(button);
    describe();
    const begin = () => {
      if (request !== version) return;
      seek(next ? next.start : 0);
      video.playbackRate = Number(speed.value);
      video.play().catch(() => {
        if (request === version) status.textContent = '已定位到所选起点，请点击视频上的播放键继续。';
      });
    };
    if (video.readyState >= 1) begin();
    else video.addEventListener('loadedmetadata', begin, { once: true });
  }
  buttons.forEach(button => {
    button.disabled = false;
    button.addEventListener('click', () => playRange({
      start: Number(button.dataset.start), end: Number(button.dataset.end),
      label: button.dataset.label, complete: false
    }, button));
  });
  fullButton.disabled = speed.disabled = false;
  fullButton.addEventListener('click', () => playRange(null, null));
  speed.addEventListener('change', () => { video.playbackRate = Number(speed.value); });
  video.addEventListener('play', () => {
    // Native Play after a segment has ended continues the original full video.
    if (selection && selection.complete) { selection = null; highlight(null); }
    stopFrame();
    tick();
  });
  video.addEventListener('pause', () => { stopFrame(); describe(); });
  video.addEventListener('timeupdate', checkEnd);
  video.addEventListener('seeking', () => {
    if (!ownSeek) { ++version; selection = null; highlight(null); stopFrame(); }
    ownSeek = false;
  });
  video.addEventListener('seeked', () => { ownSeek = false; checkEnd(); });
  video.addEventListener('ended', () => { stopFrame(); checkEnd(); });
  video.addEventListener('error', () => {
    ++version; stopFrame();
    status.textContent = `视频加载失败；请使用完整附件包中的${video.dataset.fallback}按标注时间查看。`;
  });
  describe();
}

if (typeof document !== 'undefined') {
  document.querySelectorAll('video[data-rounds]').forEach(video => {
    const mid = video.id.replace(/^video-/, '');
    setupVideoSegments(video,
      [...document.querySelectorAll(`.seek[data-video="${video.id}"]`)],
      document.querySelector(`.play-full[data-video="${video.id}"]`),
      document.getElementById(`speed-${mid}`), document.getElementById(`segment-status-${mid}`),
      JSON.parse(video.dataset.rounds), window);
  });
}
