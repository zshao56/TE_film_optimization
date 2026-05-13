
import {base,title,image,callout,foot} from './common.mjs';
export async function slide08(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Real-world benchmark');
title(slide,ctx,'FDM verification confirms strong candidates across applications','Each scenario screens 50,000 candidates and verifies 50 candidates with FDM.');
await image(slide,ctx,'fig_benchmark_best_fdm.png',70,220,555,315);
await image(slide,ctx,'fig_benchmark_errors.png',660,220,500,315);
callout(slide,ctx,105,560,430,68,'Best verified values','Battery 75.8 K; Engine 185.0 K; Phone 29.1 K; Glass 20.4 K; Skin 6.9 K.','#16a34a');
callout(slide,ctx,675,560,430,68,'Hardest case','Engine forced flat has the largest error and underprediction bias.','#dc2626');
foot(slide,ctx,8); return slide; }
