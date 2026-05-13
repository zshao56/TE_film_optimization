
const DARK = '#111827';
const MUTED = '#64748b';
const BLUE = '#2563eb';
const RED = '#dc2626';
const GREEN = '#16a34a';
const ORANGE = '#ea580c';
const BG = '#f8fafc';
const ASSET_DIR = "/Users/apple/Library/CloudStorage/OneDrive-\u4e2a\u4eba/git/TE_film/outputs/manual-20260513-173312/presentations/expanded-rebuild-v1-results/assets";

export function asset(name) { return `${ASSET_DIR}/${name}`; }

export function base(slide, ctx, section='Expanded Rebuild v1') {
  ctx.addShape(slide, {x:0,y:0,w:1280,h:720,fill:BG,line:ctx.line()});
  ctx.addText(slide, {x:54,y:34,w:520,h:24,text:section,fontSize:15,color:MUTED,bold:true});
  ctx.addShape(slide, {x:54,y:67,w:74,h:4,fill:BLUE,line:ctx.line()});
  ctx.addText(slide, {x:1080,y:34,w:146,h:24,text:'TE film surrogate',fontSize:13,color:MUTED,align:'right'});
}
export function title(slide, ctx, text, sub='') {
  ctx.addText(slide, {x:54,y:82,w:900,h:70,text,fontSize:32,color:DARK,bold:true,typeface:ctx.fonts.title});
  if (sub) ctx.addText(slide, {x:56,y:174,w:920,h:34,text:sub,fontSize:16,color:MUTED});
}
export function foot(slide, ctx, n) { ctx.addText(slide, {x:1160,y:674,w:64,h:22,text:String(n).padStart(2,'0'),fontSize:13,color:MUTED,align:'right'}); }
export function metric(slide, ctx, x, y, w, label, value, color=BLUE) {
  ctx.addShape(slide, {x,y,w,h:94,fill:'#ffffff',line:{style:'solid',fill:'#e5e7eb',width:1}});
  ctx.addText(slide, {x:x+18,y:y+16,w:w-36,h:20,text:label,fontSize:12,color:MUTED,bold:true});
  ctx.addText(slide, {x:x+18,y:y+42,w:w-36,h:36,text:value,fontSize:29,color,bold:true,typeface:ctx.fonts.title});
}
export function bullet(slide, ctx, x, y, text, color=DARK, size=17) {
  ctx.addShape(slide, {x,y:y+8,w:7,h:7,fill:BLUE,line:ctx.line()});
  ctx.addText(slide, {x:x+20,y,w:500,h:44,text,fontSize:size,color});
}
export async function image(slide, ctx, name, x, y, w, h, fit='contain') { await ctx.addImage(slide, {path: asset(name), x, y, w, h, fit}); }
export function callout(slide, ctx, x, y, w, h, head, body, color=BLUE) {
  ctx.addShape(slide, {x,y,w,h,fill:'#ffffff',line:{style:'solid',fill:'#dbeafe',width:1.3}});
  ctx.addShape(slide, {x,y,w:6,h,fill:color,line:ctx.line()});
  ctx.addText(slide, {x:x+18,y:y+16,w:w-36,h:24,text:head,fontSize:15,color:DARK,bold:true});
  ctx.addText(slide, {x:x+18,y:y+44,w:w-36,h:h-62,text:body,fontSize:12.6,color:MUTED});
}
