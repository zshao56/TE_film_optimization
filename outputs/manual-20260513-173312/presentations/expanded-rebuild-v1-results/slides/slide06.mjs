
import {base,title,image,metric,foot} from './common.mjs';
export async function slide06(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Held-out accuracy');
title(slide,ctx,'Held-out accuracy is stable across validation and test','The model generalizes across the expanded FDM distribution.');
await image(slide,ctx,'fig_eval_overall_mae_rmse.png',62,225,480,315);
await image(slide,ctx,'fig_prediction_scatter_test_enhanced.png',650,220,465,390);
metric(slide,ctx,85,565,140,'Test MAE','2.87 K','#16a34a'); metric(slide,ctx,245,565,140,'Test RMSE','3.74 K','#ea580c'); metric(slide,ctx,405,565,140,'Test R2','0.965','#2563eb');
foot(slide,ctx,6); return slide; }
