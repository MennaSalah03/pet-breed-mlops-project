import json
from pathlib import Path

import structlog

logger = structlog.getLogger()


def save_run_artifacts(
    output_dir, class_names: list, mean: list, std: list, resize=(224, 224)
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)

    transform_config = {"mean": mean, "std": std, "resize": list(resize)}
    with open(output_dir / "transform_config.json", "w") as f:
        json.dump(transform_config, f)

    logger.info("artifacts.metadata_saved", output_dir=str(output_dir))
    return output_dir
