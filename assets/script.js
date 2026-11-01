import { setupNotes } from "./notes.js";

export const siteData = {
  "news": [
    {
      "date": "Oct 2026",
      "title": "Tsinghua University Comprehensive Scholarship",
      "body": "Awarded the Tsinghua University Comprehensive Scholarship (清华大学综合奖学金)."
    },
    {
      "date": "Aug 2025",
      "title": "Joined Tsinghua University",
      "body": "I joined the Department of Computer Science and Technology at Tsinghua University."
    }
  ],
  "research": [
    {
      "id": "aaarena",
      "date": "2026",
      "status": "arXiv preprint · AAArena",
      "title": "Can AI Agents Learn Their Way to the Top? Evaluating Heuristic Learning in a Long-Running Game Agent Competition",
      "authors": "Kaisen Yang, Qingle Liu, Kejin Wang, Yicheng Zhao, Jieming Li, et al.",
      "summary": "AAArena evaluates adversarial heuristic learning across 12 games and 1,920 archived human programs. Agents interpret game rules, select opponents, analyze replays, and revise executable policies under fixed match and evaluation budgets, while keeping model weights frozen.",
      "links": [
        {
          "label": "Project",
          "href": "https://aaarena.net/"
        },
        {
          "label": "arXiv",
          "href": "https://arxiv.org/abs/2610.12341"
        }
      ]
    },
    {
      "id": "roboretry",
      "date": "2026",
      "status": "Accepted at CVPR 2026 3D-LLM/VLA Workshop · Submitted to EMNLP 2026",
      "title": "What Do VLAs Actually Learn through In-Context Failure Conditioning?",
      "authors": "Jiajun Liu, Jieming Li, Zi Zhuang, Hang Yu, Qingli Chen, Liu Cao, Yingxi Lu, Ruoqu Chen, Yuhang Cao, Chenyu Zhang, Yankai Lin, Mengdi Xu",
      "summary": "RoboRetry asks whether a vision-language-action policy can improve a retry by attending to its own failed attempt. Through controlled failure slots and experiments across 12 RLBench tasks, we separate structural retry cues from genuinely useful corrective content and introduce FailureSlot, a dataset of 1,334 naturally occurring VLA failure trajectories with human annotations.",
      "image": "assets/images/roboretry-figure.png",
      "imageAlt": "RoboRetry overview showing a VLA learning from multimodal failure context before retrying a task",
      "links": [
        {
          "label": "Workshop",
          "href": "https://openreview.net/forum?id=vpzo1dW0Zm"
        },
        {
          "label": "EMNLP",
          "href": "https://openreview.net/forum?id=3Jpz2z9MIZ"
        }
      ]
    }
  ],
  "posts": [
    {
      "slug": "wam",
      "date": "2026",
      "category": "Writing & Education",
      "title": "WAM 写作智能体｜在追问中让想法变成自己的文章",
      "summary": "从模糊灵感，到自己的观点。一个陪你提出问题、辨析概念、梳理论据的写作伙伴。",
      "tags": [
        "Rust",
        "Agents",
        "Education"
      ],
      "image": "assets/images/wam-card.png",
      "links": [
        {
          "label": "代码仓库",
          "href": "https://github.com/lijiemingjimmy/writing-agent-rust"
        },
        {
          "label": "立即体验",
          "href": "https://lijiemingjimmy.github.io/writing-coach-agent/"
        }
      ],
      "body": [
        "我有一个粗略的想法，但我应该怎么把这个想法变成一个可以写成文章的提案？WAM 从这一刻开始，陪你把模糊的兴趣推进成值得讨论的问题。",
        "它通过追问具体经历、辨析概念、梳理论据，帮助你逐步形成自己的判断与文章思路。你可以把它当做一款开放式游戏，你的选择决定讨论走向。",
        "项目以 Rust 构建核心 Agent，结合写作阶段记忆、课程资料检索与可追溯的运行过程，并提供学生对话界面和教师教学工作台。防代写机制让引导始终围绕学生自己的思考与修改展开。",
        "从一次困惑出发，看看几轮对话能把你的想法带到哪里。可联系作者微信 li-jm25 获得体验权限。"
      ]
    },
    {
      "slug": "rua-eval",
      "externalUrl": "https://openrua.ai/blog/RUAEval",
      "date": "2026",
      "category": "Research Notes",
      "title": "RUA-Eval",
      "summary": "Read the OpenRUA team’s blog on RUA-Eval.",
      "tags": [
        "OpenRUA",
        "Embodied AI",
        "Evaluation"
      ],
      "links": [
        {
          "label": "Read on OpenRUA",
          "href": "https://openrua.ai/blog/RUAEval"
        }
      ],
      "body": [
        "RUA-Eval · OpenRUA team. Read the full article on the OpenRUA website using the link above."
      ]
    }
  ]
};

const topLevelViews = new Set(["main", "info", "research", "blog", "notes"]);

export function normalizeRoute(hash = "") {
  const route = hash.replace(/^#\/?/, "").replace(/\/$/, "");
  if (route.startsWith("blog/")) {
    const slug = route.slice(5).trim();
    if (slug) return { view: "article", slug };
  }
  if (route.startsWith("info-")) return { view: "info", slug: null };
  if (topLevelViews.has(route)) return { view: route, slug: null };
  return { view: "main", slug: null };
}

export function getFilteredPosts(posts, category) {
  if (!category || category === "All") return [...posts];
  return posts.filter((post) => post.category === category);
}

const escapeHtml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

function linkMarkup(link) {
  return `<a class="text-link" href="${escapeHtml(link.href)}" target="_blank" rel="noreferrer">${escapeHtml(link.label)} <span aria-hidden="true">↗</span></a>`;
}

export function renderResearch() {
  const container = document.querySelector("#researchList");
  if (!container) return;
  container.innerHTML = siteData.research
    .map(
      (item) => `
        <article class="research-entry" id="${escapeHtml(item.id)}">
          <figure class="research-figure">
            ${item.image ? `<img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.imageAlt)}" loading="lazy" />` : `<div class="arena-art"><span>AI AGENT ARENA</span><strong>AAArena</strong><p>Learn. Play. Improve.</p><small>12 games · 1,920 human programs</small></div>`}
          </figure>
          <div class="research-copy">
            <p class="research-meta"><time>${escapeHtml(item.date)}</time><span>${escapeHtml(item.status)}</span></p>
            <h2>${escapeHtml(item.title)}</h2>
            <p class="research-authors">${escapeHtml(item.authors).replace("Jieming Li", "<strong>Jieming Li</strong>")}</p>
            <p>${escapeHtml(item.summary)}</p>
            <div class="inline-links">${item.links.map(linkMarkup).join("")}</div>
          </div>
        </article>`,
    )
    .join("");
}

let activeCategory = "All";

export function renderBlog(category = activeCategory) {
  const filters = document.querySelector("#blogCategories");
  const list = document.querySelector("#blogList");
  if (!filters || !list) return;
  activeCategory = category;
  const categories = ["All", ...new Set(siteData.posts.map((post) => post.category))];
  filters.innerHTML = categories
    .map((name) => {
      const count = name === "All" ? siteData.posts.length : getFilteredPosts(siteData.posts, name).length;
      return `<button class="category-button${name === category ? " active" : ""}" type="button" data-category="${escapeHtml(name)}"><span>${escapeHtml(name)}</span><strong>${count}</strong></button>`;
    })
    .join("");

  const posts = getFilteredPosts(siteData.posts, category);
  list.innerHTML = posts.length
    ? posts
        .map(
          (post) => `
            <article class="post-card">
              ${post.image ? `<div class="wam-banner"><img src="${escapeHtml(post.image)}" alt="WAM — 让思想，破纸而出。从模糊灵感，到自己的观点" /></div>` : ""}
              <div class="post-kicker"><span>${escapeHtml(post.category)}</span><time>${escapeHtml(post.date)}</time></div>
              <h2>${escapeHtml(post.title)}</h2>
              <p>${escapeHtml(post.summary)}</p>
              ${post.links ? `<div class="inline-links">${post.links.map(linkMarkup).join("")}</div>` : ""}
              <div class="post-footer"><div class="tag-list">${post.tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("")}</div><a class="read-link" href="${post.externalUrl ? escapeHtml(post.externalUrl) : `#blog/${escapeHtml(post.slug)}`}" ${post.externalUrl ? 'target="_blank" rel="noreferrer"' : ""}>Read <span aria-hidden="true">→</span></a></div>
            </article>`,
        )
        .join("")
    : `<p class="empty-state">No posts in this category yet.</p>`;

  filters.querySelectorAll("[data-category]").forEach((button) => {
    button.addEventListener("click", () => renderBlog(button.dataset.category));
  });
}

export function renderArticle(slug) {
  const view = document.querySelector('[data-view="article"]');
  if (!view) return false;
  const post = siteData.posts.find((item) => item.slug === slug);
  if (!post) return false;
  view.innerHTML = `
    <article class="article-shell">
      <a class="back-link" href="#blog">← Back to Blog</a>
      <p class="eyebrow">${escapeHtml(post.category)} · ${escapeHtml(post.date)}</p>
      ${post.image ? `<div class="wam-banner"><img src="${escapeHtml(post.image)}" alt="WAM — Writing Agency Mentor" /></div>` : ""}
      <h1 tabindex="-1">${escapeHtml(post.title)}</h1>
      ${post.slug === "wam" ? `<p class="wam-author">计算机系 李捷铭 · Project lead: Jieming Li</p>` : ""}
      ${post.links ? `<div class="inline-links">${post.links.map(linkMarkup).join("")}</div>` : ""}
      <p class="article-deck">${escapeHtml(post.summary)}</p>
      <div class="article-body">${post.body.map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`).join("")}</div>
      <div class="tag-list">${post.tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("")}</div>
    </article>`;
  return true;
}

let routeFromUser = false;

export function applyRoute(hash = window.location.hash) {
  let route = normalizeRoute(hash);
  if (route.view === "article" && !renderArticle(route.slug)) route = { view: "blog", slug: null };

  document.querySelectorAll("[data-view]").forEach((view) => {
    view.hidden = view.dataset.view !== route.view;
  });
  document.querySelectorAll("[data-route]").forEach((link) => {
    const active = link.dataset.route === (route.view === "article" ? "blog" : route.view);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });

  const labels = { main: "Home", info: "My Info", research: "Research", blog: "Blog", article: "Blog", notes: "Notes" };
  document.title = `${labels[route.view]} · Jieming Li`;
  document.querySelector(".site-nav")?.classList.remove("open");
  document.querySelector("#menuToggle")?.setAttribute("aria-expanded", "false");

  if (hash.startsWith("#info-")) requestAnimationFrame(() => document.getElementById(hash.slice(1))?.scrollIntoView());

  if (routeFromUser) {
    document.querySelector(`[data-view="${route.view}"] h1`)?.focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "smooth" });
    routeFromUser = false;
  }
}

function setupSite() {
  setupNotes();
  renderResearch();
  renderBlog();
  document.querySelectorAll("[data-route], .read-link, .back-link").forEach((link) => {
    link.addEventListener("click", () => {
      routeFromUser = true;
    });
  });
  window.addEventListener("hashchange", () => applyRoute());
  document.querySelector("#menuToggle")?.addEventListener("click", (event) => {
    const nav = document.querySelector(".site-nav");
    const open = nav.classList.toggle("open");
    event.currentTarget.setAttribute("aria-expanded", String(open));
  });
  applyRoute();
}

if (typeof document !== "undefined") {
  setupSite();
}
