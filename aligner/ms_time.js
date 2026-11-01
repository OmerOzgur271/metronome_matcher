// Show the audio players' current time and length to the millisecond (Gradio shows whole seconds).
//
// Gradio's waveform player moves a cursor across the waveform at an exact percentage, so
// position = that percentage x the file's exact length. The lengths come from the server in
// #audio-durations (data-durations = {"player elem_id": seconds}); players not listed keep Gradio's text.
(() => {
  const fmt = (s) => {
    const ms = Math.max(0, Math.round(s * 1000));
    return `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
  };

  const durations = () => {
    const el = document.getElementById("audio-durations");
    try {
      return JSON.parse(el?.dataset.durations || "{}");
    } catch {
      return {};
    }
  };

  const set = (el, text) => {
    if (el && el.textContent !== text) el.textContent = text;
  };

  const update = () => {
    for (const [id, length] of Object.entries(durations())) {
      const player = document.getElementById(id);
      const cursor = player?.querySelector("#waveform > div")?.shadowRoot?.querySelector(".cursor");
      if (!cursor) continue;
      const fraction = parseFloat(cursor.style.left) / 100 || 0;
      set(player.querySelector("time#time"), fmt(fraction * length));
      set(player.querySelector("time#duration"), fmt(length));
    }
  };

  // Rewrite right after Gradio writes its m:ss (before the browser paints), and every frame while playing.
  new MutationObserver(update).observe(document.documentElement, { subtree: true, childList: true, characterData: true });
  const loop = () => {
    update();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
})();
