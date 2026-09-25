Demo images for `python -m edge.detect --source samples\<file>.jpg`.

Every image is a CONSTRUCTION-SITE scene, so each demo matches the project's target
environment. They are copies of images from the VALIDATION split of datasets/ppe4;
the TEST split is deliberately left untouched so it stays a clean final measurement.

Source dataset: Roboflow Universe, "Construction Site Safety" by Roboflow Universe
Projects - https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
Licence: CC BY 4.0 (attribution required if these images are redistributed).

compliant_candidate.jpg
    source : datasets/ppe4/val/images/ppe_1228_jpg.rf.29b714c415ea74df3160f56a603463a0.jpg
    scene  : worker in red hard hat + hi-vis, outdoor site
    labels : person x1, helmet x1, vest x1

helmet_only.jpg
    source : datasets/ppe4/val/images/-3154-_png_jpg.rf.f118da2b1c20afb78ff93dc7d558f42c.jpg
    scene  : worker in yellow hard hat on timber framing, no vest
    labels : person x1, helmet x1

vest_only.jpg
    source : datasets/ppe4/val/images/youtube-470_jpg.rf.6aebf4cfb6c5a7e703c6532ca5090517.jpg
    scene  : worker in hi-vis at scaffolding, no helmet
    labels : person x1, vest x1

person_only.jpg
    source : datasets/ppe4/val/images/residential_jpg.rf.17181770c64333009b9101e1572c5112.jpg
    scene  : residential build: two workers, ladder and tools, no helmet, no vest
    labels : person x2

Expected verdicts (helmet + vest required, the default):
    compliant_candidate.jpg -> COMPLIANT
    helmet_only.jpg         -> MISSING_VEST
    vest_only.jpg           -> MISSING_HELMET
    person_only.jpg         -> MISSING_HELMET_AND_VEST
The verdict comes from what the MODEL detects, not from these labels, so a missed
detection can change it - that is the behaviour the demo is meant to show.

demo_clip.mp4
    A 12-second test clip (10 fps, 640x640) built from the four images above, three
    seconds each, by scripts/make_demo_clip.py. It exists so the video endpoint can be
    demonstrated without filming anything; it is a slideshow, NOT real site footage, and
    the report should say so. Expect the verdict to change as the clip moves from one
    scene to the next.
