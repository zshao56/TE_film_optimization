import {base,title,image,callout,foot} from './common.mjs';
export async function slide15(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Terminal definition');
title(slide,ctx,'Hot/cold terminals are measured on the top surface','The bottom face is the thermal hot boundary; Delta T_parallel is recovered between two top-surface electrode windows.');
await image(slide,ctx,'fig_hot_cold_boundary_definition.png',56,230,500,245);
await image(slide,ctx,'fig_best_hot_cold_terminal_locations.png',590,222,620,285);
callout(slide,ctx,90,586,405,80,'Boundary convention','Hot boundary: bottom face z=0. Cooling: top and side surfaces convect to ambient air.','#dc2626');
callout(slide,ctx,655,586,405,80,'Electrode convention','Hot/cold terminals are 0.5 mm x 0.5 mm windows on the top surface, chosen to maximize Delta T_parallel.','#2563eb');
foot(slide,ctx,15); return slide; }
