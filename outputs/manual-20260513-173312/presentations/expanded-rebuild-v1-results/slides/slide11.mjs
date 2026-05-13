
import {base,title,image,callout,foot} from './common.mjs';
export async function slide11(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Error analysis');
title(slide,ctx,'Where the model still struggles','The remaining weakness is concentrated in high-forced and high-Delta-T regimes.');
await image(slide,ctx,'fig_per_family_test_mae.png',70,220,500,305);
await image(slide,ctx,'fig_per_scenario_test_mae_top12.png',620,210,520,330);
callout(slide,ctx,135,555,420,70,'Family-level result','Errors are broadly stable; random-smoothed geometries are the most difficult by MAE.','#2563eb');
callout(slide,ctx,670,555,420,70,'Scenario-level result','Hardest cases cluster around strong-forced and boundary-complexity regimes.','#ea580c');
foot(slide,ctx,11); return slide; }
