const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export async function setupNotes() {
  const list = document.querySelector('#notesList'), search = document.querySelector('#notesSearch'), count = document.querySelector('#notesCount');
  if (!list || !search || !count) return;
  list.classList.add('notes-grid');
  try {
    const response = await fetch(new URL('./notes-index.json', import.meta.url));
    if (!response.ok) throw new Error('Could not load notes');
    const notes = await response.json();
    let limit = 24;
    function render() {
      const query = search.value.trim().toLocaleLowerCase();
      const results = notes.filter(note => `${note.title} ${note.filename} ${note.summary}`.toLocaleLowerCase().includes(query));
      count.textContent = `${results.length} / ${notes.length} notes`;
      list.innerHTML = results.length ? results.slice(0, limit).map(note => `<article class="note-card"><h2><a href="${escape(note.href)}">${escape(note.title)}</a></h2><p>${escape(note.summary)}</p><a class="read-link" href="${escape(note.href)}">Read note →</a></article>`).join('') : '<p>No matching notes. Try another keyword.</p>';
    }
    const more = document.createElement('button');
    more.type = 'button'; more.className = 'notes-more'; more.textContent = 'Load more notes';
    list.after(more);
    const renderPage = render;
    render = () => { renderPage(); more.hidden = list.querySelectorAll('.note-card').length >= Number(count.textContent.split(' / ')[0]); };
    more.addEventListener('click', () => { limit += 24; render(); });
    search.addEventListener('input', () => { limit = 24; render(); });
    render();
  } catch {
    count.textContent = 'Notes could not be loaded.';
    list.innerHTML = '<p>Please reload the page to try again.</p>';
  }
}
