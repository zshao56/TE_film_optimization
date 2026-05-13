
import {base,title,metric,foot,image,callout} from './common.mjs';
export async function slide01(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Main results');
title(slide,ctx,'Surrogate-assisted inverse design for thermal films','Expanded 100k FDM dataset -> 3D CNN surrogate -> FDM-verified real-world candidates');
metric(slide,ctx,58,235,202,'Test R2','0.965'); metric(slide,ctx,282,235,202,'Test MAE','2.87 K','#16a34a'); metric(slide,ctx,506,235,202,'Top-10% recall','92.2%','#7c3aed'); metric(slide,ctx,730,235,202,'Best engine FDM','185.0 K','#dc2626');
await image(slide,ctx,'fig_workflow_overview.png',70,382,1040,205);
callout(slide,ctx,930,230,260,128,'Main thesis','The surrogate is strong enough for fast screening, while FDM verification remains necessary for final performance claims.','#2563eb'); foot(slide,ctx,1); return slide; }
