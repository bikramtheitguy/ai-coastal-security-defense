// WCAG 2.x text contrast (1.4.3) for visible text, run in the page.
// axe-core leaves text over the theme's gradient as "needs review"; this
// check resolves it conservatively. Each text element is scrolled into view
// and the browser's own hit-test stack at its centre (elementsFromPoint)
// gives the boxes painted beneath it; their backgrounds are composited, a
// gradient counts every colour stop and the worst ratio wins. Disabled
// controls are exempt under 1.4.3; visually hidden text is skipped.
// Returns the failures only.
() => {
  const parse = value => {
    const match = value.match(/rgba?\(([^)]+)\)/);
    if (!match) return null;
    const parts = match[1].split(/[ ,/]+/).filter(Boolean).map(Number);
    return [parts[0], parts[1], parts[2], parts.length > 3 ? parts[3] : 1];
  };
  const channel = v => {
    v /= 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  const luminance = c => 0.2126 * channel(c[0]) + 0.7152 * channel(c[1]) + 0.0722 * channel(c[2]);
  const ratio = (a, b) => {
    const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (high + 0.05) / (low + 0.05);
  };
  const over = (top, under) => [0, 1, 2].map(i => top[i] * top[3] + under[i] * (1 - top[3])).concat([1]);
  const painted = node => {
    const style = getComputedStyle(node);
    const image = style.backgroundImage;
    const stops = image && image !== 'none' ? (image.match(/rgba?\([^)]+\)/g) || []).map(parse) : [];
    if (stops.length) return { stops, opaque: true };
    const colour = parse(style.backgroundColor);
    return colour && colour[3] > 0 ? { stops: [colour], opaque: colour[3] >= 1 } : null;
  };

  const sample = el => {
    el.scrollIntoView({ block: 'center', inline: 'nearest' });
    const box = el.getBoundingClientRect();
    const x = Math.min(Math.max(box.left + box.width / 2, 0), innerWidth - 1);
    const y = Math.min(Math.max(box.top + box.height / 2, 0), innerHeight - 1);
    const stack = document.elementsFromPoint(x, y);
    const start = stack.indexOf(el);
    // Boxes painted at or below the text, nearest first; overlays above it are ignored.
    const beneath = start >= 0 ? stack.slice(start) : stack.filter(node => node.contains(el));
    const layers = [];
    for (const node of beneath) {
      const paint = painted(node);
      if (!paint) continue;
      layers.push({ node, stops: paint.stops });
      if (paint.opaque) break;
    }
    // Opacity between the text and its nearest painted box fades the text against it.
    let opacity = 1;
    const nearest = layers.length ? layers[0].node : null;
    for (let node = el; node && node !== nearest; node = node.parentElement) {
      opacity *= Number(getComputedStyle(node).opacity);
    }
    const page = parse(getComputedStyle(document.documentElement).backgroundColor);
    let backgrounds = [page && page[3] >= 1 ? page : [255, 255, 255, 1]];
    for (const { stops } of layers.reverse()) {
      backgrounds = stops.flatMap(stop => backgrounds.map(under => (stop[3] >= 1 ? stop : over(stop, under))));
    }
    return { backgrounds, opacity };
  };

  const failures = [];
  const seen = new Set();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const el = walker.currentNode.parentElement;
    const text = walker.currentNode.textContent.trim();
    if (!el || !text || seen.has(el)) continue;
    seen.add(el);
    const style = getComputedStyle(el);
    const box = el.getBoundingClientRect();
    if (box.width <= 1 || box.height <= 1 || style.visibility === 'hidden') continue;
    // Off-canvas or clipped helpers (skip links, live regions) are not visible text.
    if (box.right <= 0 || box.left >= document.documentElement.clientWidth) continue;
    if (el.closest('[aria-hidden="true"], svg, canvas, [disabled], [aria-disabled="true"]')) continue;
    const { backgrounds, opacity } = sample(el);
    if (opacity < 0.05) continue; // visually hidden
    const raw = parse(style.color);
    const fg = [raw[0], raw[1], raw[2], raw[3] * opacity];
    const worst = Math.min(...backgrounds.map(bg => ratio(fg[3] < 1 ? over(fg, bg) : fg, bg)));
    const size = parseFloat(style.fontSize);
    const large = size >= 24 || (Number(style.fontWeight) >= 700 && size >= 18.66);
    const required = large ? 3 : 4.5;
    if (worst < required) {
      failures.push({
        text: text.slice(0, 60),
        ratio: Math.round(worst * 100) / 100,
        required,
        color: style.color,
        opacity: Math.round(opacity * 100) / 100,
        container: (el.closest('[data-testid]') || {}).dataset?.testid || '',
      });
    }
  }
  scrollTo(0, 0);
  return failures;
}
