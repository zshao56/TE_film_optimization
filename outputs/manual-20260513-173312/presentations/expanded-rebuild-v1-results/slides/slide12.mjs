
import {base,title,bullet,callout,foot} from './common.mjs';
export async function slide12(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Claims and next steps');
title(slide,ctx,'Main paper claim: neural screening plus FDM verification works','Use this run as the primary result; future work should target calibration in extreme regimes.');
bullet(slide,ctx,80,240,'Claim 1: The surrogate learns the expanded simulation distribution: test R2 = 0.965, MAE = 2.87 K.');
bullet(slide,ctx,80,310,'Claim 2: Ranking quality is high enough for inverse design: top-10% recall = 92.2%.');
bullet(slide,ctx,80,380,'Claim 3: FDM verification is necessary, especially for high-forced engine-like scenarios.');
bullet(slide,ctx,80,450,'Claim 4: The verified benchmark finds strong candidates across all five real-world scenarios.');
callout(slide,ctx,735,255,370,110,'Next experiment','Add targeted high-Delta-T / strong-forced data and test whether engine underprediction decreases.','#dc2626');
callout(slide,ctx,735,415,370,110,'Reporting rule','Report verified FDM performance as final. Use surrogate metrics to justify the screening stage.','#16a34a');
foot(slide,ctx,12); return slide; }
