
import tensorrt as trt
import torch
import numpy as np

class TRTModule(torch.nn.Module):
    def __init__(self, engine_path, device='cuda'):
        super().__init__()
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING) # Less verbose
        
        print(f"🚀 Loading TRT Engine: {engine_path}")
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if not self.engine:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
            
        self.context = self.engine.create_execution_context()
        
        # Inspect IO (TensorRT 10.x API)
        self.input_name = None
        self.output_name = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
                
        if not self.input_name or not self.output_name:
             raise RuntimeError("Could not identify input/output tensors in engine")

    def forward(self, x):
        """
        x: Torch tensor (Batch, Channels, Height, Width) on GPU
        """
        # Ensure input is contiguous
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Set dynamic shape
        self.context.set_input_shape(self.input_name, x.shape)
        
        # Prepare output buffer
        # Output shape depends on batch size. (B, 100) for ResNet JNR
        batch_size = x.shape[0]
        # We assume output is (B, 100) or we can query output dims if fixed?
        # But output dim 0 is also dynamic.
        # We allocate (B, 100) float32.
        # Ideally query the engine for output dim logic, but hardcoding for JNR is faster/safer.
        output = torch.empty((batch_size, 100), dtype=torch.float32, device=self.device)
        
        # Bind
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, output.data_ptr())
        
        # Execute
        stream = torch.cuda.current_stream(device=self.device)
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        
        return output
