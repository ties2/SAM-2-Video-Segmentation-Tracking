import io
from typing import Any, Protocol

import joblib


# Custom Exceptions for Loading
class ModelLoadError(Exception):
    pass


class UnsupportedModelFormat(Exception):
    pass


class InvalidModelArtifact(Exception):
    pass


# Abstraction (Interface)
class ModelLoader(Protocol):
    """
    Interface for different model loaders.
    """

    def load(self, artifact_bytes: bytes) -> Any: ...


# Implementation for Scikit-Learn + Joblib
class SklearnJoblibLoader:
    def load(self, artifact_bytes: bytes) -> Any:
        try:
            # Convert bytes to an in-memory virtual file (file-like object)
            virtual_file = io.BytesIO(artifact_bytes)

            # Load the model from the virtual file
            model = joblib.load(virtual_file)

            # Validation: Is this actually a Scikit-Learn model?
            # (Machine learning models usually have 'predict' or 'transform' methods)
            if not hasattr(model, "predict") and not hasattr(model, "transform"):
                raise InvalidModelArtifact(
                    "The loaded object is not a valid scikit-learn model (missing 'predict' or 'transform' methods)."
                )

            return model

        except InvalidModelArtifact:
            raise
        except Exception as e:
            # Handle cases where the file is corrupted or invalid
            raise ModelLoadError(f"Failed to load scikit-learn model: {e!s}") from e


# Implementation for SAM 2 / SAM 2.1 PyTorch checkpoints (video object segmentation)
class Sam2CheckpointLoader:
    """Turn a SAM 2 checkpoint (bytes) into a ready-to-use ``Sam2VideoSegmenter``.

    * ``torch.load(weights_only=True)`` → no arbitrary code execution from a tampered file.
    * The variant (tiny/small/base+/large) is inferred from tensor shapes, so the
      registry does not need an extra "config" column.
    * Heavy imports (torch, sam2) happen here, lazily, so the platform still starts and
      serves scikit-learn models on machines without a deep-learning stack.
    """

    def __init__(self, device: str = "cpu", num_threads: int | None = None):
        self.device = device
        self.num_threads = num_threads

    def load(self, artifact_bytes: bytes) -> Any:
        try:
            from src.vos.predictor import Sam2Config, Sam2VideoSegmenter, load_state_dict
        except ImportError as e:  # torch / sam2 not installed in this image
            raise ModelLoadError(f"SAM 2 runtime not installed: {e!s}") from e
        try:
            state_dict = load_state_dict(artifact_bytes)
        except Exception as e:
            raise InvalidModelArtifact(f"Not a valid SAM 2 checkpoint: {e!s}") from e
        try:
            cfg = Sam2Config(device=self.device, num_threads=self.num_threads)
            return Sam2VideoSegmenter(cfg, state_dict=state_dict)
        except ValueError as e:
            raise InvalidModelArtifact(str(e)) from e
        except Exception as e:
            raise ModelLoadError(f"Failed to build SAM 2 predictor: {e!s}") from e


# Registry: (framework, model_format) -> loader factory.
# Adding a new runtime (ONNX, TensorRT, …) is one entry here, not an edit of the service layer.
def _sam2_factory() -> ModelLoader:
    from src.serving.config import settings

    return Sam2CheckpointLoader(device=settings.vos_device, num_threads=settings.vos_num_threads)


LOADER_REGISTRY: dict[tuple[str, str], Any] = {
    ("scikit-learn", "joblib"): SklearnJoblibLoader,
    ("sam2", "pytorch-checkpoint"): _sam2_factory,
}

# what each framework is able to do (used to reject e.g. /predict on a video model)
TASK_BY_FRAMEWORK = {"scikit-learn": "tabular-prediction", "sam2": "video-segmentation"}


def get_loader(framework: str, model_format: str) -> ModelLoader:
    factory = LOADER_REGISTRY.get((framework, model_format))
    if factory is None:
        supported = ", ".join(f"{f}/{m}" for f, m in LOADER_REGISTRY)
        raise UnsupportedModelFormat(f"Unsupported format {framework}/{model_format} (supported: {supported})")
    return factory()
