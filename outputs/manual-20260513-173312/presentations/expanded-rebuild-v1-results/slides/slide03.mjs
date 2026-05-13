
import {base,title,image,foot,callout} from './common.mjs';
export async function slide03(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Method');
title(slide,ctx,'End-to-end pipeline','The workflow separates fast surrogate screening from physically verified final claims.');
await image(slide,ctx,'fig_workflow_overview_detailed.png',66,216,1105,335);
callout(slide,ctx,92,580,470,86,'Model input','Material mask + hot-boundary map + 26 scalar physics descriptors.','#7c3aed');
callout(slide,ctx,690,580,420,86,'Decision rule','Surrogate ranks candidates; FDM supplies reported Delta T.','#16a34a');
foot(slide,ctx,3); return slide; }
