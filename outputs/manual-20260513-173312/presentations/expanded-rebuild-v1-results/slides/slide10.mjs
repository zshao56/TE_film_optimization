
import {base,title,image,foot} from './common.mjs';
export async function slide10(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Screening distributions');
title(slide,ctx,'Candidate distributions show the full screening funnel','The surrogate narrows 50,000 candidates to a top-50 verification set per scenario.');
await image(slide,ctx,'fig_real_world_screening_distributions_wide.png',70,225,1040,405);
foot(slide,ctx,10); return slide; }
