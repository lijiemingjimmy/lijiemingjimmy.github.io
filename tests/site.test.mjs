import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { getFilteredPosts, normalizeRoute, siteData } from "../assets/script.js";

const root = new URL("../", import.meta.url);
const read = (path) => readFile(new URL(path, root), "utf8");

test("normalizes top-level and blog article routes", () => {
  assert.deepEqual(normalizeRoute("#research"), { view: "research", slug: null });
  assert.deepEqual(normalizeRoute("#blog/learning-from-failure"), {
    view: "article",
    slug: "learning-from-failure",
  });
  assert.deepEqual(normalizeRoute("#unknown"), { view: "main", slug: null });
});

test("filters blog posts without mutating the source collection", () => {
  const original = [...siteData.posts];
  const research = getFilteredPosts(siteData.posts, "Research Notes");
  assert.ok(research.length > 0);
  assert.ok(research.every((post) => post.category === "Research Notes"));
  assert.deepEqual(siteData.posts, original);
});

test("publishes accurate research and news metadata", () => {
  const roboRetry = siteData.research.find((item) => item.id === "roboretry");
  assert.ok(roboRetry);
  assert.match(roboRetry.status, /CVPR 2026.*3D-LLM\/VLA/i);
  assert.match(roboRetry.status, /Submitted to EMNLP 2026/i);
  assert.match(roboRetry.authors, /Jiajun Liu, Jieming Li/);
  assert.ok(roboRetry.links.some((link) => link.href.includes("vpzo1dW0Zm")));
  assert.ok(roboRetry.links.some((link) => link.href.includes("3Jpz2z9MIZ")));

  assert.deepEqual(siteData.news[0], {
    date: "Aug 2025",
    title: "Joined Tsinghua University",
    body: "I joined the Department of Computer Science and Technology at Tsinghua University.",
  });
});

test("HTML contains four navigable views and comprehensive profile content", async () => {
  const html = await read("index.html");
  for (const view of ["main", "info", "research", "blog", "article"]) {
    assert.match(html, new RegExp(`data-view=["']${view}["']`));
  }
  assert.match(html, /Institute for Interdisciplinary Information Sciences/);
  assert.match(html, /A strangely balanced athlete/);
  assert.match(html, /best ultimate player on the badminton team/i);
  assert.match(html, /id=["']researchList["']/);
  assert.match(html, /id=["']blogList["']/);
  assert.doesNotMatch(html, /hello@example\.com/);
});

test("homepage uses the requested Jimmy branding and portrait", async () => {
  const html = await read("index.html");
  assert.match(html, /lijiemingjimmy\.github\.io/);
  assert.match(html, /Welcome to Jimmy's homepage/);
  assert.match(html, /assets\/images\/profile\.jpg/);
  assert.doesNotMatch(html, /class=["']brand["']/);
  assert.doesNotMatch(html, /Learning how the world changes/);
  assert.doesNotMatch(html, /Jieming\.li · Embodied intelligence/);
});

test("CSS provides view routing, publication rows, responsiveness and accessibility", async () => {
  const css = await read("assets/styles.css");
  assert.match(css, /\.view\[hidden\]/);
  assert.match(css, /\.research-entry\s*\{[^}]*display:\s*grid/s);
  assert.match(css, /@media\s*\(max-width:\s*720px\)/);
  assert.match(css, /:focus-visible/);
  assert.match(css, /prefers-reduced-motion/);
});

test("script exposes research, blog and routing renderers", async () => {
  const script = await read("assets/script.js");
  for (const name of ["renderResearch", "renderBlog", "renderArticle", "applyRoute"]) {
    assert.match(script, new RegExp(`function\\s+${name}\\b`));
  }
  assert.match(script, /document\.title\s*=/);
  assert.match(script, /aria-current/);
  assert.match(script, /No posts in this category yet/);
});
