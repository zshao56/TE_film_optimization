
import {base,title,image,callout,foot} from './common.mjs';
export async function slide07(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Ranking quality');
title(slide,ctx,'The surrogate is strongest as a ranking model','Top-candidate discovery is the relevant objective for inverse design.');
await image(slide,ctx,'fig_ranking_quality.png',70,220,500,310);
await image(slide,ctx,'fig_top_region_error_bias.png',640,220,500,310);
callout(slide,ctx,105,555,430,70,'Discovery result','Test top-10% precision and recall are both 0.922; Spearman rank correlation is 0.986.','#16a34a');
callout(slide,ctx,675,555,430,70,'Calibration note','The top high-Delta-T region has negative bias around -3.7 K, so FDM verification remains necessary.','#dc2626');
foot(slide,ctx,7); return slide; }
