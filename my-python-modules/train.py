import torch

from engine import train_one_epoch, evaluate

# Importing python modules
from common.manage_log import *
from common.utils import * 

def training_model(parameters, model, device, data_loader_train, data_loader_valid):
  
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=parameters['neural_network_model']['learning_rate'],
                                momentum=parameters['neural_network_model']['momentum'],
                                weight_decay=parameters['neural_network_model']['weight_decay'])

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=parameters['neural_network_model']['gamma'],
        gamma=parameters['neural_network_model']['gamma']
    )

    # training for 10 epochs
    # num_epochs = WHITE_MOLD_EPOCHS

    losses_per_epoch = []
    train_loss_list_excel = []

    # training model 
    for epoch in range(parameters['neural_network_model']['number_epochs']):
        # training for one epoch
        metric_logger_return = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)

        # logging_info(f'metric_logger_return: {metric_logger_return}')
        # logging_info(f'metric_logger_return.loss: {metric_logger_return.loss}')
        # logging_info(f'metric_logger_return.loss: {type(metric_logger_return.loss)}')
        # logging_info(f'metric_logger_return.loss.median: {metric_logger_return.loss.median}')
        # logging_info(f'metric_logger_return.loss: {metric_logger_return.loss}')
        # losses_per_epoch.append(metric_logger_return("loss"))

        losses_per_epoch.append(metric_logger_return.loss.median)
        train_loss_list_excel.append([epoch+1, metric_logger_return.loss.median])

        logging_info(f'losses_per_epoch: {losses_per_epoch}')        

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_evaluator_return = evaluate(model, data_loader_valid, device=device)


    # plot loss function for training
    path_and_filename = os.path.join(
        parameters['training_results']['metrics_folder'],     
        parameters['neural_network_model']['model_name'] + \
                    '_train_loss.png'
    )
    title = f'Training Loss for model {parameters["neural_network_model"]["model_name"]}'
    x_label = "Epochs"
    y_label = "Train Loss"
    logging_info(f'path_and_filename: {path_and_filename}')
    Utils.save_plot(losses_per_epoch, path_and_filename, title, x_label, y_label)

    # saving loss list to excel file
    logging_info(f'train_loss_list_excel final: {train_loss_list_excel}')    
    path_and_filename = os.path.join(
        parameters['training_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + '_train_loss.xlsx'
    )
    logging_info(f'path_and_filename: {path_and_filename}')

    Utils.save_losses(train_loss_list_excel, path_and_filename)          

    # returning model trained
    return model
