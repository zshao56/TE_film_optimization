
import {base,title,image,foot,callout} from './common.mjs';
export async function slide05(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Training');
title(slide,ctx,'Training converged with best checkpoint at epoch 73','The final evaluation uses the validation-best model, not the last epoch.');
await image(slide,ctx,'fig_training_curve.png',70,215,780,385);
callout(slide,ctx,890,220,270,110,'Best checkpoint','Epoch 73\nValidation loss 0.0337\nLearning rate 0.000150','#16a34a');
callout(slide,ctx,890,370,270,120,'Interpretation','Later epochs reduce train loss but do not consistently improve validation loss; checkpoint selection avoids using an overfit final epoch.','#2563eb');
foot(slide,ctx,5); return slide; }
