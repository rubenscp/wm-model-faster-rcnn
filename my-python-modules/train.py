import torch

from engine import train_one_epoch, evaluate


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

    # training model 
    for epoch in range(parameters['neural_network_model']['number_epochs']):
        # training for one epoch
        metric_logger_return = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_evaluator_return = evaluate(model, data_loader_valid, device=device)

    # returning model trained
    return model
