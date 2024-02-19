import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as torchtrans

# Importing python modules
from manage_log import *
from visualization import * 

# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')


def inference_faster_rcnn_model_with_dataset_test(parameters, model, device, dataset_test):

    # processing test images

    # detection_threshold = 0.5 DO NOT USE

    logging_info(f'Test dataset size #4: {len(dataset_test)}')

    # logging_info(f'Threshold confidence: {WHITE_MOLD_THRESHOLD}')

    for i in range(len(dataset_test)):

        # logging_info(f'dataset_test[i]: {dataset_test[i]}')
        # logging_info(f'dataset_test[i].img: {dataset_test[i].imgs}')

        # pick one image from the test set
        img, target = dataset_test[i]
        image_name = dataset_test.imgs[i]

        logging_info(f'')
        logging_info(f'-'*100)
        logging_info(f'Inference of image #{i+1} - {image_name}')

        # put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])[0]

        # Move all values from GPU to CPU
        for key, value in prediction.items():
            prediction[key] = value.cpu()

        # setting results folder
        local_results_faster_rcnn_predict_path = os.path.join(
            parameters['inference_results']['inferenced_image_folder']
        )
        # logging_info(f'')
        # logging_info('EXPECTED OUTPUT\n')
        path_and_filename_expected_image = os.path.join(local_results_faster_rcnn_predict_path, image_name + '_expected.jpg')
        plot_img_bbox(torch_to_pil(img), target, path_and_filename_image=path_and_filename_expected_image)

        # logging_info('FASTER R-CNN MODEL OUTPUT\n')
        # nms_prediction = apply_nms(prediction, iou_thresh=0.01)
        nms_prediction = apply_nms(prediction, iou_thresh=parameters['neural_network_model']['iou_threshold'])
        
        path_and_filename_predicted_image = os.path.join(local_results_faster_rcnn_predict_path, image_name + '_predicted.jpg')
        plot_img_bbox(
            torch_to_pil(img), 
            nms_prediction, 
            threshold=parameters['neural_network_model']['threshold'], 
            path_and_filename_image=path_and_filename_predicted_image,
            classes=parameters['neural_network_model']['classes']
        )

        logging_info(f'nms_prediction {nms_prediction}')