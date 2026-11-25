// Event-level checks without a browser: original timestamps and player behavior.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, 'report.html'), 'utf8');
const trace = JSON.parse(fs.readFileSync(path.join(__dirname, 'data/direct-trial03-timeline.json')));
const context = vm.createContext({});
const script = fs.readFileSync(path.join(__dirname, 'video-segments.js'), 'utf8');
assert.ok(html.includes(script), 'Built HTML must embed the current player implementation');
vm.runInContext(script, context);
class Target {
  constructor(dataset = {}) { this.dataset = dataset; this.listeners = {}; this.attrs = {}; this.disabled = true; }
  addEventListener(type, fn, options = {}) { (this.listeners[type] ||= []).push({ fn, once: options.once }); }
  emit(type) {
    for (const item of [...(this.listeners[type] || [])]) {
      if (item.once) this.listeners[type] = this.listeners[type].filter(x => x !== item);
      item.fn();
    }
  }
  setAttribute(key, value) { this.attrs[key] = value; }
}
class Video extends Target {
  constructor() { super(); this.paused = true; this.readyState = 1; this.time = 0; this.seeking = false; }
  get currentTime() { return this.time; }
  set currentTime(value) {
    this.seeking = true; this.time = value; this.emit('seeking');
    this.seeking = false; this.emit('seeked');
  }
  pause() { this.paused = true; this.emit('pause'); }
  play() { this.paused = false; this.emit('play'); return Promise.resolve(); }
  advance(value) { this.time = value; this.emit('timeupdate'); }
}
const tags = [...html.matchAll(/<button[^>]*class="seek"[^>]*>/g)].map(x => x[0]).filter(tag => tag.includes('data-video="video-direct-astra"')).slice(0, 5);
const buttons = tags.map(tag => new Target(Object.fromEntries(
  [...tag.matchAll(/data-(start|end|label)="([^"]*)"/g)].map(x => [x[1], x[2]])
)));
assert.equal(buttons.length, 5);
const expected = [6, 7, 8, 10, 13];
buttons.forEach((b, i) => {
  const row = trace[expected[i] - 1];
  assert.equal(Number(b.dataset.start), row.start_step / 20);
  assert.equal(Number(b.dataset.end), row.end_step / 20);
});
const video = new Video(), full = new Target(), speed = new Target(), status = new Target();
speed.value = '1';
const clock = { requestAnimationFrame: () => 1, cancelAnimationFrame: () => {} };
context.setupVideoSegments(video, buttons, full, speed, status,
  trace.map(r => ({ call: r.call, start: r.start_step / 20, end: r.end_step / 20 })), clock);
for (const b of buttons) {
  b.emit('click');
  assert.equal(video.currentTime, Number(b.dataset.start));
  assert.equal(video.paused, false);
  assert.equal(b.attrs['aria-pressed'], 'true');
  assert.ok(status.textContent.includes(b.dataset.label));
  video.advance(Number(b.dataset.end) + 0.08);
  assert.equal(video.paused, true);
  assert.equal(video.currentTime, Number(b.dataset.end));
  assert.ok(status.textContent.includes('本轮播放结束'));
}
video.play(); // Native play after a chapter end must not immediately stop again.
video.advance(22.38);
assert.equal(video.paused, false);
assert.ok(status.textContent.includes('完整视频模式'));
buttons[0].emit('click');
video.currentTime = 18; // Manual scrubbing exits bounded segment playback.
video.advance(19);
assert.equal(video.paused, false);
assert.ok(buttons.every(b => b.attrs['aria-pressed'] === 'false'));
speed.value = '0.5'; speed.emit('change');
assert.equal(video.playbackRate, 0.5);
full.emit('click');
assert.equal(video.currentTime, 0);
video.advance(8);
assert.equal(video.paused, false);
assert.ok(status.textContent.includes('第8轮'));
video.readyState = 0;
buttons[0].emit('click'); buttons[1].emit('click');
video.readyState = 1; video.emit('loadedmetadata');
assert.equal(video.currentTime, 6.5); // Only the latest selection starts after metadata arrives.
video.advance(7.11);
assert.equal(video.currentTime, 7.1);
assert.equal(video.paused, true);
console.log('PASS: 5 timestamp mappings, segment playback/end pause, full playback, manual seek, speed, metadata race. Browser playback not exercised.');

const players = [];
for (const [id, count] of [['direct-astra', 13], ['direct-qwen', 19], ['direct-boundary', 10]]) {
  const timeline = JSON.parse(fs.readFileSync(path.join(__dirname, `data/${id}-timeline.json`)));
  assert.equal(timeline.length, count);
  const tags = [...html.matchAll(/<button[^>]*class="seek"[^>]*>/g)].map(x => x[0]).filter(t => t.includes(`data-video="video-${id}"`));
  const buttons = tags.map(t => new Target(Object.fromEntries([...t.matchAll(/data-(start|end|label)="([^"]*)"/g)].map(x => [x[1], x[2]]))));
  assert.equal(buttons.length, count + 5);
  const video = new Video(), full = new Target(), speed = new Target(), status = new Target(); speed.value = '1';
  context.setupVideoSegments(video, buttons, full, speed, status, timeline.map(r => ({call:r.call,start:r.start_step/20,end:r.end_step/20})), clock);
  for (const row of timeline) {
    const b = buttons.find(b => b.dataset.label.startsWith(`第${row.call}轮 ·`));
    assert.equal(+b.dataset.start, row.start_step/20); assert.equal(+b.dataset.end, row.end_step/20);
    b.emit('click'); assert.equal(video.currentTime, +b.dataset.start); assert.equal(video.paused, false);
    video.advance(+b.dataset.end + .06); assert.equal(video.currentTime, +b.dataset.end); assert.equal(video.paused, true);
    assert.ok(buttons.filter(x => x.dataset.start === b.dataset.start).every(x => x.attrs['aria-pressed']==='true'));
  }
  players.push({video, buttons});
}
players[0].buttons[0].emit('click');
assert.ok(players.slice(1).every(p => p.video.paused), 'Video controls must remain independent');
const qwen = JSON.parse(fs.readFileSync(path.join(__dirname, 'data/direct-qwen-timeline.json')));
assert.equal(qwen[18].tool, 'give_up'); assert.equal(qwen[18].start_step, 620); assert.equal(qwen[18].end_step, 621);
console.log('PASS: all 42 rounds, 57 buttons, duplicate highlights, independent players, Qwen termination boundary.');
