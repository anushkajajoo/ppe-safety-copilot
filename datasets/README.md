Datasets live here but are NOT committed to Git (see .gitignore).

- datasets/ppe4/  <- created by:
      python -m training.prepare_ppe4 --src <downloaded YOLOv8 export .zip or folder>
  Primary dataset: a small seeded subset (~1000/250/250) of the Roboflow Universe
  "Construction Site Safety" dataset (CC BY 4.0). See docs/decisions.md D-010.

- Alternative (larger): python -m training.prepare_sh17 --src <SH17 folder>   (see D-002)

Record the download date and source URL in docs/decisions.md.
