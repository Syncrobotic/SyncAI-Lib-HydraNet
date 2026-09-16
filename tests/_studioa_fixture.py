"""Small source-bound StudioA annotation packages for data contract tests."""

import numpy as np
from PIL import Image

from syncai_hydranet.data.studioa_autolabel import Candidate, annotate
from syncai_hydranet.data.studioa_review import digest, write_json


def source_package(root):
    (root / "frames").mkdir(parents=True)
    (root / "images").mkdir()
    frames = []
    for store in ("Kaohsiung", "Taichung", "Tao-Hsin"):
        for index, split in enumerate(("train", "val", "test"), 1):
            identity = f"{len(frames):04d}-{store}-cam{index:02d}"
            name = f"images/{identity}.png"
            Image.new("RGB", (40, 32), color=(len(frames) * 20, 50, 80)).save(root / name)
            frames.append(
                {
                    "id": identity,
                    "image": name,
                    "image_sha256": digest(root / name),
                    "store": store,
                    "camera": f"{store}-cam{index:02d}",
                    "original_split": split,
                    "image_size_px": [40, 32],
                }
            )
    write_json(root / "job.json", {"frames": frames})
    outputs = {}
    for frame in frames:
        mask = np.zeros((32, 40), dtype=bool)
        mask[:24, :30] = True
        data = annotate([Candidate("floor", mask, 0.9)], mask.shape)
        data.update(
            frame_id=frame["id"],
            job_sha256=digest(root / "job.json"),
            image_sha256=frame["image_sha256"],
        )
        name = f"frames/{frame['id']}.json"
        write_json(root / name, data)
        outputs[name] = digest(root / name)
    write_json(
        root / "report.json",
        {"status": "completed", "outputs": outputs, "job_sha256": digest(root / "job.json")},
    )
    return frames
