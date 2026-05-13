import {base,title,image,callout,foot} from './common.mjs';
export async function slide14(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Verified structures');
title(slide,ctx,'Best verified candidates can be inspected as scenario-specific 3D structures','Each panel shows the FDM-best design found after surrogate screening and top-candidate verification.');
await image(slide,ctx,'fig_scenario_best_3d_structures_slide.png',52,215,1180,315);
callout(slide,ctx,120,590,470,80,'Structure layer','Colored voxels show the high-conductivity material mask selected by inverse design.','#2563eb');
callout(slide,ctx,700,590,420,80,'Boundary layer','The top surface shows the real-world hot-boundary map.','#ea580c');
foot(slide,ctx,14); return slide; }
