"""Command-line entry points, registered as console scripts in pyproject.toml.

hydranet-train           train a model
hydranet-eval            run validation on a checkpoint
hydranet-infer-image     overlay predictions on an image or folder
hydranet-infer-video     overlay predictions on a video
hydranet-scene           project the commissioned scene through its camera
hydranet-export-onnx     export for TensorRT
hydranet-prepare-ade20k  filter ADE20K down to an indoor subset
hydranet-prepare-cocostuff prepare COCO-Stuff (mind the off-by-one PNG ids)
hydranet-annotation      validate annotation taxonomies
hydranet-report          summarise and compare finished runs

Ten, which is what `[project.scripts]` in pyproject.toml declares.
"""
