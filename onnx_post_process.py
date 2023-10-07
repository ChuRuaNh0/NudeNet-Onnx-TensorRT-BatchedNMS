import torch
import onnx 
# from models.experimental import attempt_load
import argparse
import onnx_graphsurgeon as gs 
from onnx import shape_inference
import torch.nn as nn

class NudeNetPostProcess(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = model
        # self.model.eval()

    def forward(self, output):
        """ 
            Split output [n_batch, 84, n_bboxes] to 3 output: bboxes, scores, classes
        """ 
        # output = self.model(input)
        # output = output[0]
        # print
        output = output.permute(0, 2, 1)
        print(output[0][0])
        print("[INFO] Output's origin model shape: ", output.shape)
        bboxes_x = output[..., 0:1]
        bboxes_y = output[..., 1:2]
        bboxes_w = output[..., 2:3]
        bboxes_h = output[..., 3:4]
        bboxes_x1 = bboxes_x - bboxes_w/2
        bboxes_y1 = bboxes_y - bboxes_h/2
        bboxes_x2 = bboxes_x + bboxes_w/2
        bboxes_y2 = bboxes_y + bboxes_h/2
        bboxes = torch.cat([bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2], dim = -1)
        bboxes = bboxes.unsqueeze(2) # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        obj_conf = output[..., 4:]
        scores = obj_conf
        return bboxes, scores
    
if __name__ == "__main__":
    model = NudeNetPostProcess()
    model.eval()
    logit = torch.rand(size = (1, 22 , 2100))
    boxes_tensor, confs_tensor = model(logit)
    # Export the model
    torch.onnx.export(model,               # model being run
                      (logit,),                         # model input (or a tuple for multiple inputs)
                      "post_process.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=17,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['bboxes', 'scores'], # the model's output names
                      dynamic_axes={ 'input': {0: 'batch_size'},
                                    'bboxes' : [0, 1], 'scores' : [0, 1]})
    # assert False

    model1 = onnx.load("/data/disk1/hungpham/NudeNet/nudenet/best.onnx")
    model2 = onnx.load("post_process.onnx")


    combined_model = onnx.compose.merge_models(
        model1, model2,
        io_map=[('output0', 'input')]
                )

    onnx.checker.check_model(combined_model)
    graph = gs.import_onnx(combined_model)
    onnx.save(gs.export_onnx(graph), "nude_post_process_v2.onnx")