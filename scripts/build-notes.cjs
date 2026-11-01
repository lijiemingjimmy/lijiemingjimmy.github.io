// npm install --prefix /tmp/notes-build markdown-it@15.0.3 markdown-it-texmath@1.0.0 katex@0.19.1 mermaid@12.2.0
// NOTES_DEPENDENCIES=/tmp/notes-build/node_modules node scripts/build-notes.cjs [Ideas directory]
const fs=require('fs'), path=require('path');
const deps=process.env.NOTES_DEPENDENCIES||'/tmp/notes-build/node_modules';
const MarkdownIt=require(path.join(deps,'markdown-it')), texmath=require(path.join(deps,'markdown-it-texmath')), katex=require(path.join(deps,'katex'));
const root=path.resolve(__dirname,'..'), source=path.resolve(process.argv[2]||path.join(root,'..')), output=path.join(root,'notes');
fs.mkdirSync(output,{recursive:true}); fs.mkdirSync(path.join(output,'vendor'),{recursive:true});
fs.mkdirSync(path.join(output,'vendor/katex'),{recursive:true});
fs.cpSync(path.join(deps,'katex/dist/fonts'),path.join(output,'vendor/katex/fonts'),{recursive:true});
fs.copyFileSync(path.join(deps,'katex/dist/katex.min.css'),path.join(output,'vendor/katex/katex.min.css'));
fs.copyFileSync(path.join(deps,'mermaid/dist/mermaid.min.js'),path.join(output,'vendor/mermaid.min.js'));
fs.copyFileSync(path.join(deps,'mermaid/LICENSE'),path.join(output,'vendor/MERMAID-LICENSE'));
fs.copyFileSync(path.join(deps,'katex/LICENSE'),path.join(output,'vendor/katex/LICENSE'));
const files=fs.readdirSync(source).filter(f=>f.endsWith('.md')).sort();
const mapping=new Map(files.map(f=>[f,f.replace(/\.md$/,'.html')]));
const esc=s=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const md=new MarkdownIt({html:false,linkify:true,typographer:false}).use(texmath,{engine:katex,delimiters:['dollars','brackets'],katexOptions:{strict:'ignore',trust:false}});
const fence=md.renderer.rules.fence;
md.renderer.rules.fence=(tokens,i,opts,env,self)=>tokens[i].info.trim()==='mermaid'?`<figure class="diagram"><pre class="mermaid">${esc(tokens[i].content)}</pre><details><summary>Diagram source</summary><pre>${esc(tokens[i].content)}</pre></details></figure>`:fence(tokens,i,opts,env,self);
let toc=[]; md.renderer.rules.heading_open=(tokens,i,opts,env,self)=>{const title=tokens[i+1].content;const id='section-'+toc.length;tokens[i].attrSet('id',id);if(tokens[i].tag!=='h1')toc.push({id,title,level:tokens[i].tag});else tokens[i].attrSet('id','noteTitle');return self.renderToken(tokens,i,opts);};
function rewrite(tokens){for(let i=0;i<tokens.length;i++){const t=tokens[i];if(t.children)rewrite(t.children);for(const attr of ['href','src']){let u=t.attrGet(attr);if(!u||/^https?:|^mailto:|^#|^data:/.test(u))continue;let decoded=decodeURIComponent(u); const [filename,fragment]=decoded.split('#');const base=path.basename(filename);if(mapping.has(base)){t.attrSet(attr,encodeURI(mapping.get(base))+(fragment?'#'+encodeURIComponent(fragment):''));}
else if(attr==='src'&&!filename.startsWith('http')){const actual=path.isAbsolute(filename)?filename:path.resolve(source,filename);if(fs.existsSync(actual)&&actual.startsWith(source+path.sep)){const dest=path.join(output,'media',path.relative(source,actual));fs.mkdirSync(path.dirname(dest),{recursive:true});fs.copyFileSync(actual,dest);t.attrSet(attr,encodeURI(path.relative(output,dest)));}}
else if(filename.startsWith('/Users/')||(!filename.startsWith('/')&&!fs.existsSync(path.resolve(source,filename)))){if(t.type==='link_open'){t.tag='span';t.attrs=[['class','unavailable-source'],['title','本地参考资料，未在网站公开']];let depth=1;for(let j=i+1;j<tokens.length;j++){if(tokens[j].type==='link_open')depth++;if(tokens[j].type==='link_close'&&!--depth){tokens[j].tag='span';break;}}}}}}}
const index=[];
for(const file of files){let text=fs.readFileSync(path.join(source,file),'utf8');
// Restore unmistakable escaped-command corruption in the source note, without editing it.
if(file==='I-BFM-人形交互策略.md') text=text.replaceAll('\x07pprox','\\approx').replaceAll('\top','\\top').replaceAll('\nho','\\rho');const title=(text.match(/^#\s+(.+)$/m)||[])[1]||file.slice(0,-3);toc=[];const tokens=md.parse(text,{});rewrite(tokens);const content=md.renderer.render(tokens,md.options,{});const outline=toc.filter(x=>x.level==='h2').map(x=>`<li><a href="#${x.id}">${esc(x.title)}</a></li>`).join('');
const html=`<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${esc(title)} · Jieming Li</title><link rel="stylesheet" href="../assets/styles.css"><link rel="stylesheet" href="../assets/notes.css"><link rel="stylesheet" href="vendor/katex/katex.min.css"></head><body class="note-page"><header class="note-nav"><a href="../index.html#blog">← Blog</a><a href="../index.html">Jieming Li</a></header><main class="note-layout"><aside class="note-outline"><p class="eyebrow">On this page</p><nav aria-label="目录"><ol>${outline}</ol></nav></aside><article class="note-content" id="content"><p class="eyebrow">Research notebook · 研究笔记</p>${content}<footer><a href="../index.html#blog">← Back to Blog</a></footer></article></main>${text.includes('```mermaid')?'<script src="vendor/mermaid.min.js"></script><script>mermaid.initialize({startOnLoad:true,securityLevel:"strict",theme:"neutral",fontFamily:"system-ui",flowchart:{htmlLabels:false}});</script>':''}</body></html>`;
fs.writeFileSync(path.join(output,mapping.get(file)),html);const description=text.split('\n').filter(x=>x.trim()&&!/^#|^\||^[-*]|^来源|^```|^\$/.test(x)).slice(0,2).join(' ').replace(/\[([^\]]+)\]\([^)]*\)/g,'$1').replace(/[*`]/g,'').slice(0,180);
index.push({title,filename:file,href:'notes/'+encodeURI(mapping.get(file)),summary:description});}
fs.writeFileSync(path.join(root,'assets/notes-index.json'),JSON.stringify(index,null,2)+'\n');
console.log(`Rendered ${files.length} notes.`);
