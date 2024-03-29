import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as torchtrans

# Importing python modules
from visualization import * 

from common.manage_log import *
from common.metrics import *

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

    threshold = parameters['neural_network_model']['threshold']

    # creating metric object 
    inference_metric = Metrics(model=parameters['neural_network_model']['model_name'])

    for i in range(len(dataset_test)):

        # logging_info(f'dataset_test[i]: {dataset_test[i]}')
        # logging_info(f'dataset_test[i].img: {dataset_test[i].imgs}')

        # pick one image from the test set
        img, target = dataset_test[i]
        image_name = dataset_test.imgs[i]

        # logging_info(f'')
        # logging_info(f'-'*100)
        logging_info(f'Test image #{i+1} - {image_name}')

        # put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])[0]

        # Move all values from GPU to CPU
        for key, value in prediction.items():
            prediction[key] = value.cpu()      

        # setting results folder
        local_results_faster_rcnn_predict_path = os.path.join(
            parameters['test_results']['inferenced_image_folder']
        )
        # logging_info(f'')
        # logging_info('EXPECTED OUTPUT\n')
        path_and_filename_expected_image = os.path.join(local_results_faster_rcnn_predict_path, image_name + '_expected.jpg')
        plot_img_bbox(torch_to_pil(img), target, path_and_filename_image=path_and_filename_expected_image)

        # logging_info('FASTER R-CNN MODEL OUTPUT\n')
        # nms_prediction = apply_nms(prediction, iou_thresh=0.01)
        nms_prediction = apply_nms(prediction, iou_thresh=parameters['neural_network_model']['non_maximum_suppression'])
        
        path_and_filename_predicted_image = os.path.join(local_results_faster_rcnn_predict_path, image_name + '_predicted.jpg')
        plot_img_bbox(
            torch_to_pil(img), 
            nms_prediction, 
            threshold=parameters['neural_network_model']['threshold'], 
            path_and_filename_image=path_and_filename_predicted_image,
            classes=parameters['neural_network_model']['classes']
        )

        # logging_info(f'nms_prediction {nms_prediction}')

        # select just the predictions with score greater than confidence threshold
        logging_info(f'nms_prediction: {nms_prediction}')
        valid_prediction = {
            'boxes': [],
            'labels': [],
            'scores': [],
        }
        for i, score in enumerate(nms_prediction['scores']):
            logging_info(f'i: {i}   score: {score}   threshold: {threshold}')
            if score >= threshold:
                valid_prediction['boxes'].append((nms_prediction['boxes'][i]).numpy())
                valid_prediction['labels'].append((nms_prediction['labels'][i]).squeeze())
                valid_prediction['scores'].append((nms_prediction['scores'][i]).squeeze())

        logging_info(f'prediction: {valid_prediction}')
        logging_info(f"prediction['boxes']: {valid_prediction['boxes']}")
        logging_info(f"prediction['labels']: {valid_prediction['labels']}")
        logging_info(f"prediction['scores']: {valid_prediction['scores']}")
        valid_prediction['boxes'] = torch.tensor(valid_prediction['boxes'])
        valid_prediction['labels'] = torch.tensor(valid_prediction['labels'])
        valid_prediction['scores'] = torch.tensor(valid_prediction['scores'])
        logging_info(f'prediction converted to tensor: {valid_prediction}')


        # setting target and predicted bounding boxes for metrics 
        new_targets = []
        item_target = {
            "boxes": target['boxes'],
            "labels": target['labels']
            }
        new_targets.append(item_target)

        new_predicteds = []
        item_predicted = {
            "boxes": valid_prediction['boxes'],
            "scores": valid_prediction['scores'],
            "labels": valid_prediction['labels'],
            }
        new_predicteds.append(item_predicted)
        # print(f'--------------------------------------------------')
        # print(f'    target: {target}')
        # print(f'new_targets: {new_targets}')
        # print(f'new_predicteds: {new_predicteds}')
        # print(f'--------------------------------------------------')

        # setting target and predicted bounding boxes for metrics
        inference_metric.set_details_of_inferenced_image(
            image_name, new_targets, new_predicteds) 
        # inference_metric.target.extend(new_targets)
        # inference_metric.preds.extend(new_predicteds)
        # print(f'inference_metric.to_string: {inference_metric.to_string()}')
        # print(f'--------------------------------------------------')

    # classes 
    classes = parameters['neural_network_model']['classes']

    # Computing Confusion Matrix 
    model_name = parameters['neural_network_model']['model_name']
    num_classes = 5
    threshold = parameters['neural_network_model']['threshold']
    iou_threshold = parameters['neural_network_model']['iou_threshold']
    metrics_folder = parameters['test_results']['metrics_folder']
    inference_metric.compute_confusion_matrix(model_name, num_classes, threshold, iou_threshold, metrics_folder)
    inference_metric.confusion_matrix_to_string()

    # saving confusion matrix plots
    title =  'Full Confusion Matrix' + \
             ' - Model: ' + parameters['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + '  # bounding box -' + \
             ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
             '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
             '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_full.png'
    )
    cm_classes = classes[0:5]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    x_labels_names.append('Ghost predictions')    
    y_labels_names.append('Undetected objects')
    format='.0f'
    Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_full.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix,
                                      path_and_filename, 
                                      x_labels_names, y_labels_names)                                    

    title = 'Confusion Matrix'
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix.png'
    )
    cm_classes = classes[1:5]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    format='.0f'
    Utils.save_plot_confusion_matrix(inference_metric.confusion_matrix, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.confusion_matrix,
                                      path_and_filename,
                                      x_labels_names, y_labels_names)                      

    title =  'Full Confusion Matrix Normalized' + \
             ' - Model: ' + parameters['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + '  # bounding box -' + \
             ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
             '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
             '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_full_normalized.png'
    )
    cm_classes = classes[0:5]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    x_labels_names.append('Ghost predictions')    
    y_labels_names.append('Undetected objects')
    format='.2f'
    Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix_normalized, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_full_normalized.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix_normalized,
                                      path_and_filename,
                                      x_labels_names, y_labels_names)                      

    title =  'Confusion Matrix Normalized' + \
             ' - Model: ' + parameters['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + '  # bounding box -' + \
             ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
             '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
             '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_normalized.png'
    )
    cm_classes = classes[1:5]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    format='.2f'
    Utils.save_plot_confusion_matrix(inference_metric.confusion_matrix_normalized, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_normalized.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.confusion_matrix_normalized,
                                      path_and_filename,
                                      x_labels_names, y_labels_names)

    # saving metrics from confusion matrix
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_metrics.xlsx'
    )
    # Utils.save_metrics_from_confusion_matrix_excel(
    #     path_and_filename, 
    #     parameters['neural_network_model']['model_name'],
    #     inference_metric.get_model_accuracy(),
    #     inference_metric.get_model_precision(),
    #     inference_metric.get_model_recall(),
    #     inference_metric.get_model_f1_score(),
    #     inference_metric.get_model_specificity(),
    #     inference_metric.get_model_dice(),
    #     inference_metric.confusion_matrix_summary["number_of_images"],
    #     inference_metric.confusion_matrix_summary["number_of_bounding_boxes_target"],
    #     inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted"],
    #     inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"],
    #     inference_metric.confusion_matrix_summary["number_of_ghost_predictions"],
    #     inference_metric.confusion_matrix_summary["number_of_undetected_objects"],
    # )

    # get performance metrics 
    # logging_info(f'')
    # logging_info(f"Performance Metrics of model {parameters['neural_network_model']['model_name']}")
    # logging_info(f'')
    # model_accuracy = inference_metric.get_model_accuracy()
    # logging_info(f'accuracy    : {model_accuracy:.4f}')
    # model_precision = inference_metric.get_model_precision()
    # logging_info(f'precision   : {model_precision:.4f}')
    # model_recall = inference_metric.get_model_recall()
    # logging_info(f'recall      : {model_recall:.4f}')
    # model_f1_score = inference_metric.get_model_f1_score()
    # logging_info(f'f1-score    : {model_f1_score:.4f}')
    # model_specificity = inference_metric.get_model_specificity()
    # logging_info(f'specificity : {model_specificity:.4f}')
    # model_dice = inference_metric.get_model_dice()
    # logging_info(f'dice        : {model_dice:.4f}')
    # logging_info(f'')

    sheet_name='summary_metrics'
    sheet_list = []
    sheet_list.append(['Metrics Results calculated by application', ''])
    sheet_list.append(['', ''])
    sheet_list.append(['Model', f'{ parameters["neural_network_model"]["model_name"]}'])
    sheet_list.append(['', ''])

    # computing TP, FP, FN and TN from confusion matrix 
    sheet_list.append(['TP / FP / FN / TN', ''])
    sheet_list.append(['TP per classes', inference_metric.tp_per_classes])
    sheet_list.append(['FP per classes', inference_metric.fp_per_classes])
    sheet_list.append(['FN per classes', inference_metric.fn_per_classes])
    sheet_list.append(['TN per classes', inference_metric.tn_per_classes])
    sheet_list.append(['', ''])
    sheet_list.append(['TP', inference_metric.tp_model])
    sheet_list.append(['FP', inference_metric.fp_model])
    sheet_list.append(['FN', inference_metric.fn_model])
    sheet_list.append(['TN', inference_metric.tn_model])
    sheet_list.append(['', ''])

    # metric measures 
    sheet_list.append(['Metric measures', ''])
    # sheet_list.append(['Accuracy', f'{inference_metric.get_model_accuracy():.8f}'])
    sheet_list.append(['Precision', f'{inference_metric.get_model_precision():.8f}'])
    sheet_list.append(['Recall', f'{inference_metric.get_model_recall():.8f}'])
    sheet_list.append(['F1-score', f'{inference_metric.get_model_f1_score():.8f}'])
    sheet_list.append(['Dice', f'{inference_metric.get_model_dice():.8f}'])
    sheet_list.append(['', ''])
    # sheet_list.append(['inference_metric.confusion_matrix_summary', inference_metric.confusion_matrix_summary])
    sheet_list.append(['number_of_images', f'{inference_metric.confusion_matrix_summary["number_of_images"]:.0f}'])
    sheet_list.append(['number_of_bounding_boxes_target', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_target"]:.0f}'])
    sheet_list.append(['number_of_bounding_boxes_predicted', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted"]:.0f}'])
    sheet_list.append(['number_of_bounding_boxes_predicted_with_target', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"]:.0f}'])
    sheet_list.append(['number_of_ghost_predictions', f'{inference_metric.confusion_matrix_summary["number_of_ghost_predictions"]:.0f}'])
    sheet_list.append(['number_of_undetected_objects', f'{inference_metric.confusion_matrix_summary["number_of_undetected_objects"]:.0f}'])

    # saving metrics sheet
    Utils.save_metrics_excel(path_and_filename, sheet_name, sheet_list)
    logging_sheet(sheet_list)
