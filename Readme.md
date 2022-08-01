# defect_detection_3d_openvino

Steps to train and run

Prerequisites

Python* 3
PyTorch* 0.4.1
OpenVINO™ 2019 R1 with Python API

Installation
cat requirements.txt | xargs -n 1 -L 1 pip3 install
pip3 install -e .

# train the pytorch model #
python3 tools/train.py(assumes data in ./data/coco in coco format, if it is not present convert it in that format, sample utility script to convert from IVI input data --> convert_to_coco.py)

#sample inference run #
python3 tools/demo.py --dataset coco_2017_val  --ckpt <path to trained model> --mean_pixel 102.9801 115.9465 122.7717 --fit_window 800 1333  --images <path_to_image> --delay 1  --show_fps  pytorch  --model segmentoly.rcnn.model_zoo.resnet_fpn_mask_rcnn.ResNet50FPNMaskRCNN --show_flops


# onnx conversion #
python3 tools/convert_to_onnx.py \
    --model segmentoly.rcnn.model_zoo.resnet_fpn_mask_rcnn.ResNet50FPNMaskRCNN \
    --ckpt <path to trained model> \
    --input_size 800 1344 \
    --dataset coco_2017_val \
    --show_flops \
    --output_file onnx_export/<path to onnx file>

# openvino conversion #
python3 <path>/mo.py \ #openvino model optimizer
    --framework onnx \
    --input_model ./onnx_export/<onnx file> \
    --output_dir ./open_vino \
    --input "im_data,im_info" \
    --output "boxes,scores,classes,batch_ids,raw_masks" \
    --mean_values "im_data[102.9801,115.9465,122.7717],im_info[0,0,0]"

# run server #
##API##
python3 getDimensions.py --dataset coco_2017_val --ckpt ./open_vino/<open vino bin file> --fit_window 800 1333 --images <dummy path, image taken from client request> --delay 1 --show_fps  openvino --model ./open_vino/<open vino xml file> -l <path to so extension file>


