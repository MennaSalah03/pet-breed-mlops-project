import os

import mlflow
import mlflow.pyfunc
import onnxruntime as ort


class PetBreedPyfunc(mlflow.pyfunc.PythonModel):
    """Thin wrapper whose main job is to package the three files as one model."""

    def load_context(self, context):
        # Retrieve the directory path from the artifacts dictionary
        artifact_dir = context.artifacts["model_artifacts"]
        
        # Construct the full path to the ONNX file
        onnx_path = os.path.join(artifact_dir, "model.onnx")
        
        # ONNXRuntime will load model.onnx and automatically detect model.onnx.data next to it
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, context, model_input):
        # expects a preprocessed float32 batch, returns logits
        return self.session.run(None, {self.input_name: model_input})[0]