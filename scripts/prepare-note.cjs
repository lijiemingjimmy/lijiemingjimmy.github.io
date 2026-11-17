// Publication edits only: retain the complete technical body and source files.
function prepareNote(text) {
  text = text.replace(/^\*\*资料状态[^\n]*\n\n/gm, '');
  text = text.replace(/^附件包含正文以外的网页界面与会话配置残留[^\n]*\n/gm, '');
  const split = text.search(/^## /m);
  let intro = split < 0 ? text : text.slice(0, split);
  const body = split < 0 ? '' : text.slice(split);
  intro = intro.replace(/(?:本文|这份笔记|依据用户提供)[^。\n]*(?:用户提供|你提供|所附解读|提供的解读|依据附件|用户附件)[^。\n]*。/g, '');
  intro = intro.replace(/(?:本文)?核对日期[：为][^。\n]*。/g, '');
  intro = intro.replace(/本次(?:通过搜索|项目站未能读取|访问 Hugging Face)[^。\n]*。/g, '');
  intro = intro.replace(/项目首页采用动态加载，[^。\n]*。/g, '');
  intro = intro.replace(/附件尾部[^。\n]*。/g, '');
  intro = intro.replace('附件标注作者为', '作者：');
  intro = intro.replace('用户提供的第三讲 PDF', '第三讲 PDF');
  intro = intro.replace('资料来自用户提供的', '资料来自');
  return (intro + body).replace(/\n{3,}/g, '\n\n');
}
module.exports = { prepareNote };
