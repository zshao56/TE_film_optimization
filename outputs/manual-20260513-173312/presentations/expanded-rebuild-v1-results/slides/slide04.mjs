
import {base,title,metric,bullet,foot,callout} from './common.mjs';
export async function slide04(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Dataset and model');
title(slide,ctx,'The expanded dataset covers the target physics space','Balanced train/val/test splits and explicit high-performance weighting.');
metric(slide,ctx,70,215,190,'Train','80,000'); metric(slide,ctx,285,215,190,'Validation','10,000'); metric(slide,ctx,500,215,190,'Test','10,000'); metric(slide,ctx,715,215,190,'Target std','19.71 K');
bullet(slide,ctx,82,360,'2 input channels: material mask and hot-boundary temperature channel.');
bullet(slide,ctx,82,425,'26 scalar descriptors encode thickness, conductivities, convection, curvature, and boundary parameters.');
bullet(slide,ctx,82,490,'Top-region weighted loss: Delta T >= 47.26 K receives weight 3.');
callout(slide,ctx,770,390,360,130,'Why this matters','Non-uniform hot boundaries and curved substrates cannot be represented by one scalar temperature alone; the field channel preserves spatial boundary information.','#2563eb');
foot(slide,ctx,4); return slide; }
