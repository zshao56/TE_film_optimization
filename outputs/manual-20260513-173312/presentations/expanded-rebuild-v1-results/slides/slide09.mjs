
import {base,title,image,callout,foot} from './common.mjs';
export async function slide09(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Verification behavior');
title(slide,ctx,'The best FDM candidates stay inside the surrogate top-50','This validates surrogate screening even when absolute calibration is imperfect.');
await image(slide,ctx,'fig_verified_candidates_calibration_and_rank.png',64,215,760,395);
callout(slide,ctx,880,210,270,112,'Positive result','Best FDM candidate rank: Battery 4, Skin 7, Glass 5, Engine 20, Phone 36.','#16a34a');
callout(slide,ctx,880,365,270,130,'Risk to report','Do not report surrogate-predicted Delta T as final performance. The final number should be verified FDM.','#dc2626');
foot(slide,ctx,9); return slide; }
