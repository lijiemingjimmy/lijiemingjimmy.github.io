export const siteData = {
  news: [
    {
      date: "Aug 2025",
      title: "Joined Tsinghua University",
      body: "I joined the Department of Computer Science and Technology at Tsinghua University.",
    },
  ],
  research: [
    {
      id: "roboretry",
      date: "2026",
      status: "Accepted at CVPR 2026 3D-LLM/VLA Workshop · Submitted to EMNLP 2026",
      title: "What Do VLAs Actually Learn through In-Context Failure Conditioning?",
      authors:
        "Jiajun Liu, Jieming Li, Zi Zhuang, Hang Yu, Qingli Chen, Liu Cao, Yingxi Lu, Ruoqu Chen, Yuhang Cao, Chenyu Zhang, Yankai Lin, Mengdi Xu",
      summary:
        "RoboRetry asks whether a vision-language-action policy can improve a retry by attending to its own failed attempt. Through controlled failure slots and experiments across 12 RLBench tasks, we separate structural retry cues from genuinely useful corrective content and introduce FailureSlot, a dataset of 1,334 naturally occurring VLA failure trajectories with human annotations.",
      image: "assets/images/roboretry-figure.png",
      imageAlt: "RoboRetry overview showing a VLA learning from multimodal failure context before retrying a task",
      links: [
        { label: "Workshop", href: "https://openreview.net/forum?id=vpzo1dW0Zm" },
        { label: "EMNLP", href: "https://openreview.net/forum?id=3Jpz2z9MIZ" },
      ],
    },
    {
      id: "writing-coach",
      date: "2026",
      status: "Tsinghua University Language Teaching Center · Educational AI",
      title: "Writing & Communication Coach Agent",
      authors: "Project lead: Jieming Li",
      summary:
        "A process-oriented writing coach that turns vague writing difficulties into diagnosable problems and concrete revision tasks. The system combines stateful Socratic dialogue, course-grounded retrieval, targeted feedback, revision tracking, and anti-ghostwriting safeguards, while helping instructors see recurring student needs.",
      image: "assets/images/writing-coach-student.jpg",
      imageAlt: "Student interface of the Writing and Communication Coach Agent",
      links: [
        { label: "Student Demo", href: "https://lijiemingjimmy.github.io/writing-coach-agent/" },
        { label: "GitHub", href: "https://github.com/lijiemingjimmy/writing-coach-agent" },
      ],
    },
  ],
  posts: [
    {
      slug: "learning-from-failure",
      date: "Jul 2026",
      category: "Research Notes",
      title: "What can a VLA learn from its own failures?",
      summary: "Notes behind RoboRetry: failure context, corrective language, and why repeating an attempt is not the same as learning from it.",
      tags: ["VLA", "Robot Learning", "RoboRetry"],
      body: [
        "A capable robot policy should not repeat the same mistake forever. Yet most vision-language-action policies treat every retry almost as a fresh start: the previous trajectory disappears, along with the evidence of what went wrong.",
        "RoboRetry turns this gap into a controlled question. We place multimodal evidence from a failed attempt directly into the policy context and ask which parts actually change the next action. The experiments distinguish the mere presence of a retry slot from useful content inside it, especially corrective language that identifies the failed target, progress, or contact decision.",
        "The broader lesson is that memory is only valuable when the policy can ground it back into the current task. A stored failure can signal that something should change; reliable recovery still requires understanding precisely what should change and how.",
      ],
    },
    {
      slug: "process-oriented-writing-agent",
      date: "Jul 2026",
      category: "Writing & Education",
      title: "Building an AI writing coach that does not write for the student",
      summary: "Design notes on diagnosis, Socratic follow-up, course-grounded feedback, and anti-ghostwriting boundaries.",
      tags: ["Agents", "Education", "Writing"],
      body: [
        "A useful writing coach should improve the student's decisions, not quietly replace the student. That principle changes the product from a text generator into a process-oriented diagnostic system.",
        "The Writing & Communication Coach first checks whether it has enough context: the assignment, the student's text, the intended claim, and the current difficulty. It then locates problems in the student's own wording, explains why they matter, and turns feedback into a small next revision task.",
        "Course materials, rubrics, and examples ground the feedback. Stateful dialogue lets the system pause for missing information and continue from the same diagnostic state. Guardrails prevent complete rewrites or submission-ready answers, preserving authorship while still making the next step concrete.",
      ],
    },
  ],
};

const topLevelViews = new Set(["main", "info", "research", "blog"]);

export function normalizeRoute(hash = "") {
  const route = hash.replace(/^#\/?/, "").replace(/\/$/, "");
  if (route.startsWith("blog/")) {
    const slug = route.slice(5).trim();
    if (slug) return { view: "article", slug };
  }
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
            <img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.imageAlt)}" loading="lazy" />
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
              <div class="post-kicker"><span>${escapeHtml(post.category)}</span><time>${escapeHtml(post.date)}</time></div>
              <h2>${escapeHtml(post.title)}</h2>
              <p>${escapeHtml(post.summary)}</p>
              <div class="post-footer"><div class="tag-list">${post.tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("")}</div><a class="read-link" href="#blog/${escapeHtml(post.slug)}">Read <span aria-hidden="true">→</span></a></div>
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
      <h1>${escapeHtml(post.title)}</h1>
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

  const labels = { main: "Home", info: "My Info", research: "Research", blog: "Blog", article: "Blog" };
  document.title = `${labels[route.view]} · Jieming Li`;
  document.querySelector(".site-nav")?.classList.remove("open");
  document.querySelector("#menuToggle")?.setAttribute("aria-expanded", "false");

  if (routeFromUser) {
    document.querySelector(`[data-view="${route.view}"] h1`)?.focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "smooth" });
    routeFromUser = false;
  }
}

function setupSite() {
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
