
import {base,title,bullet,foot,callout} from './common.mjs';
export async function slide02(presentation, ctx) { const slide=presentation.slides.add(); base(slide,ctx,'Motivation');
title(slide,ctx,'Why a surrogate is needed','Direct FDM is reliable but too slow for broad inverse design over geometry, material, and boundary-condition space.');
bullet(slide,ctx,76,230,'FDM verifies one design at a time; inverse design needs tens of thousands of candidates.');
bullet(slide,ctx,76,300,'Thermal-film performance depends jointly on 3D geometry, material contrast, convection, curvature, and hot-boundary maps.');
bullet(slide,ctx,76,370,'The optimization objective is not only low prediction error; it is reliable discovery of high-Delta-T candidates.');
callout(slide,ctx,730,220,390,105,'Research question','Can a learned surrogate rank high-performance designs well enough to guide FDM verification?', '#2563eb');
callout(slide,ctx,730,365,390,118,'Design principle','Use the neural model for screening and ranking. Use FDM for the final verified result.', '#16a34a');
foot(slide,ctx,2); return slide; }
