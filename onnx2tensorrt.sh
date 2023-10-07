/usr/src/tensorrt/bin/trtexec --onnx=<your_output_model_name>-nms.onnx \
                                --saveEngine=<your_output_trt_model_name>.trt \
                                --explicitBatch \
                                --minShapes=images:1x3x320x320 \
                                --optShapes=images:1x3x320x320 \
                                --maxShapes=images:1x3x320x320 \
                                --verbose \
                                --device=1