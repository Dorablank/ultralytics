# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import IS_JETSON, LOGGER, PYTHON_VERSION
from ultralytics.utils.checks import check_requirements, check_tensorrt, check_version

from .base import BaseBackend


class TensorRTBackend(BaseBackend):
    """NVIDIA TensorRT inference backend for GPU-accelerated deployment.

    Loads and runs inference with NVIDIA TensorRT serialized engines (.engine files). Supports both TensorRT 7-9 and
    TensorRT 10/11 APIs, dynamic input shapes, FP16 precision, DLA core offloading, and CUDA Graph inference.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load an NVIDIA TensorRT engine from a serialized .engine file.

        Args:
            weight (str | Path): Path to the .engine file with optional embedded metadata.
        """
        LOGGER.info(f"Loading {weight} for TensorRT inference...")

        if IS_JETSON and check_version(PYTHON_VERSION, "<=3.8.10"):
            check_requirements("numpy==1.23.5")

        try:
            import tensorrt as trt
        except ImportError:
            check_tensorrt()
            import tensorrt as trt

        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.2.0", msg="https://github.com/ultralytics/ultralytics/pull/24367")

        if self.device.type == "cpu":
            self.device = torch.device("cuda:0")

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)

        # Read engine file
        with open(weight, "rb") as f, trt.Runtime(logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
                dla = metadata.get("dla", None)
                if dla is not None:
                    runtime.DLA_core = int(dla)
            except UnicodeDecodeError:
                f.seek(0)
                metadata = None
            engine = runtime.deserialize_cuda_engine(f.read())
            self.apply_metadata(metadata)
        try:
            self.context = engine.create_execution_context()
        except Exception:
            LOGGER.error("TensorRT model exported with a different version than expected\n")
            raise

        # Setup bindings
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        self.dynamic = False
        # TensorRT 10 and 11 both drop the legacy binding API in favor of named I/O tensors
        self.is_trt10 = not hasattr(engine, "num_bindings")
        num = range(engine.num_io_tensors) if self.is_trt10 else range(engine.num_bindings)

        for i in num:
            if self.is_trt10:
                name = engine.get_tensor_name(i)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(engine.get_tensor_shape(name))
                profile_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2]) if is_input else None
            else:
                name = engine.get_binding_name(i)
                dtype = trt.nptype(engine.get_binding_dtype(i))
                is_input = engine.binding_is_input(i)
                shape = tuple(engine.get_binding_shape(i))
                profile_shape = tuple(engine.get_profile_shape(0, i)[1]) if is_input else None

            if is_input:
                if -1 in shape:
                    self.dynamic = True
                    if self.is_trt10:
                        self.context.set_input_shape(name, profile_shape)
                    else:
                        self.context.set_binding_shape(i, profile_shape)
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)

            shape = (
                tuple(self.context.get_tensor_shape(name))
                if self.is_trt10
                else tuple(self.context.get_binding_shape(i))
            )
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.cuda_graph = None
        self.cuda_graph_shape = None
        self.cuda_graph_stream = None
        self.cuda_graph_warmup_stream = None
        self.model = engine

    def _set_shape(self, im: torch.Tensor, persistent_input: bool = False) -> None:
        """Set a dynamic input shape and resize output buffers."""
        binding = self.bindings["images"]
        shape_changed = self.dynamic and im.shape != binding.shape
        if shape_changed:
            if self.is_trt10:
                self.context.set_input_shape("images", im.shape)
            else:
                self.context.set_binding_shape(self.model.get_binding_index("images"), im.shape)
            binding = binding._replace(shape=im.shape)

        if persistent_input and binding.data.shape != im.shape:
            data = torch.empty(im.shape, dtype=binding.data.dtype, device=self.device)
            binding = binding._replace(shape=im.shape, data=data, ptr=int(data.data_ptr()))
        self.bindings["images"] = binding
        self.binding_addrs["images"] = binding.ptr

        if not shape_changed:
            return

        self.cuda_graph = None
        for name in self.output_names:
            i = None if self.is_trt10 else self.model.get_binding_index(name)
            shape = tuple(self.context.get_tensor_shape(name) if self.is_trt10 else self.context.get_binding_shape(i))
            data = self.bindings[name].data
            data.resize_(shape)
            self.bindings[name] = self.bindings[name]._replace(shape=shape, data=data, ptr=int(data.data_ptr()))
            self.binding_addrs[name] = self.bindings[name].ptr

    def _execute_async(self, stream: torch.cuda.Stream) -> None:
        """Enqueue TensorRT inference on a CUDA stream."""
        if self.is_trt10:
            for name, address in self.binding_addrs.items():
                self.context.set_tensor_address(name, address)
            success = self.context.execute_async_v3(stream.cuda_stream)
        else:
            success = self.context.execute_async_v2(
                bindings=list(self.binding_addrs.values()), stream_handle=stream.cuda_stream
            )
        if not success:
            raise RuntimeError("TensorRT inference execution failed")

    def _capture_cuda_graph(self, shape: torch.Size) -> None:
        """Warm up and capture TensorRT inference for a fixed input shape."""
        self.cuda_graph_stream = torch.cuda.Stream(device=self.device)
        self.cuda_graph_warmup_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(self.cuda_graph_warmup_stream):
            for _ in range(3):
                self._execute_async(self.cuda_graph_warmup_stream)
        self.cuda_graph_warmup_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.cuda_graph_stream):
            self._execute_async(self.cuda_graph_stream)
        self.cuda_graph = graph
        self.cuda_graph_shape = tuple(shape)

    def _forward_cuda_graph(self, im: torch.Tensor) -> list[torch.Tensor]:
        """Run TensorRT with persistent buffers and CUDA Graph replay."""
        self._set_shape(im, persistent_input=True)
        s = self.bindings["images"].shape
        assert im.shape == s, f"input size {im.shape} not equal to engine size {s}"

        if self.cuda_graph is None or self.cuda_graph_shape != tuple(im.shape):
            self._capture_cuda_graph(im.shape)

        with torch.cuda.stream(self.cuda_graph_stream):
            self.bindings["images"].data.copy_(im)
            self.cuda_graph.replay()
        self.cuda_graph_stream.synchronize()
        return [self.bindings[x].data for x in sorted(self.output_names)]

    def forward(self, im: torch.Tensor, cuda_graph: bool = False) -> list[torch.Tensor]:
        """Run NVIDIA TensorRT inference with dynamic shape handling.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format on the CUDA device.
            cuda_graph (bool): Run inference by replaying a captured CUDA Graph.

        Returns:
            (list[torch.Tensor]): Model predictions as a list of tensors on the CUDA device.
        """
        if cuda_graph:
            return self._forward_cuda_graph(im)

        self._set_shape(im)
        s = self.bindings["images"].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"

        self.binding_addrs["images"] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return [self.bindings[x].data for x in sorted(self.output_names)]
