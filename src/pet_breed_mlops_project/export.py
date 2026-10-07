from pathlib import Path

import structlog
import torch

logger = structlog.getLogger()


def export_to_onnx(
    model, save_path, input_size: tuple = (1, 3, 224, 224), opset: int = 17
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to("cpu").eval()
    dummy_input = torch.randn(*input_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=opset,
    )

    logger.info("export.onnx.complete", path=str(save_path), opset=opset)
    return save_path
