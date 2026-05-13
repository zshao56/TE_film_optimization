import {base,title,image,callout,foot} from './common.mjs';
export async function slide13(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Relative calibration');
title(slide,ctx,'Normalize prediction error by FDM magnitude','Relative deviation changes how we interpret small-Delta-T and large-Delta-T scenarios.');
await image(slide,ctx,'fig_benchmark_relative_deviation.png',62,212,620,310);
await image(slide,ctx,'fig_benchmark_signed_relative_bias.png',705,222,500,292);
callout(slide,ctx,84,555,330,98,'Key readout','Engine MAE is 26.6 K, but relative MAE is 16.6% because FDM Delta T is large.','#dc2626');
callout(slide,ctx,458,555,330,98,'Small-target sensitivity','Skin MAE is 2.7 K, but relative MAE is 73.3% because FDM Delta T is small.','#16a34a');
callout(slide,ctx,832,555,300,98,'Reporting implication','Report both Kelvin error and normalized error for real-world benchmarks.','#2563eb');
foot(slide,ctx,13); return slide; }
