# Field Pilot Protocol (Ghana)

## Objective

Estimate real-world accuracy of the installed app against expert labelling, and
collect new Ghana field photos for dataset growth.

## Setup

1. Install the preview APK on 5-10 farmer/extension-agent phones (Android 7+).
2. One trained enumerator (or an extension agent) accompanies users for the first session.
3. Print this sheet for reference; the app works fully offline.

## Protocol

1. Each participant captures 10-20 leaves (different plants, fields, times of day).
2. For every scan, an expert (plant virologist/extension officer) independently records
   their own label for the same leaf. Labels: common_rust, gray_leaf_spot,
   northern_leaf_blight, healthy, other.
3. Record: app prediction + confidence (from the History tab), expert label, location.

## Analysis

- Build a comparison table app-vs-expert; report accuracy and confusion matrix.
- Flag every disagreement; inspect the image to decide whether the app or the expert
  was right (or both wrong).
- Add "expert-corrected" images to the local collection -> `data/raw/local/` with
  `leaf_id = <expertid>_<plantid>`, re-run the data tasks, and optionally fine-tune.

## Feedback loop

Disagreements with good image quality are the highest-value training data. Send them
(Contribute tab -> share) to the researcher with the expert label.
