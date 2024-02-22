"""
Institution: Institute of Computing - University of Campinas (IC/Unicamp)
Project: White Mold 
Description: Implements the Faster RCNN neural network model for step of training.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 06/02/2024
Version: 1.0
This implementation is based on this notebook: https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch
"""

# Basic python and ML Libraries
import os
# import datetime
from datetime import datetime

import shutil
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
# import torchvision
from torchvision import transforms as torchtrans
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
# import utils
import transforms as T

# # for image augmentations
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2

# Importing python modules
from manage_log import *
from model import *
from dataset import *
from train import * 
from tasks import Tasks

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'
NEW_FILE = True

# ###########################################
# Application Methods
# ###########################################

# ###########################################
# Methods of Level 1
# ###########################################

def main():
    """
    Main method that perform training of the neural network model.

    All values of the parameters used here are defined in the external file "wm_model_faster_rcnn_parameters.json".

    """

    # creating Tasks object 
    processing_tasks = Tasks()

    # setting dictionary initial parameters for processing
    full_path_project = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-faster-rcnn'

    # getting application parameters 
    processing_tasks.start_task('Getting application parameters')
    parameters_filename = 'wm_model_faster_rcnn_parameters.json'
    parameters = get_parameters(full_path_project, parameters_filename)
    processing_tasks.finish_task('Getting application parameters')

    # setting new values of parameters according of initial parameters
    processing_tasks.start_task('Setting input image folders')
    set_input_image_folders(parameters)
    processing_tasks.finish_task('Setting input image folders')

    # getting last running id
    processing_tasks.start_task('Getting running id')
    running_id = get_running_id(parameters)
    processing_tasks.finish_task('Getting running id')

    # setting output folder results
    processing_tasks.start_task('Setting result folders')
    set_results_folder(parameters)
    processing_tasks.finish_task('Setting result folders')
    
    # creating log file 
    processing_tasks.start_task('Creating log file')
    logging_create_log(
        parameters['training_results']['log_folder'], parameters['training_results']['log_filename']
    )
    processing_tasks.finish_task('Creating log file')
    
    logging_info('White Mold Research')
    logging_info('Training the model Faster RCNN' + LINE_FEED)

    logging_info(f'')
    logging_info(f'>> Set input image folders')
    logging_info(f'')
    logging_info(f'>> Get running id')
    logging_info(f'running id: {str(running_id)}')   
    logging_info(f'')
    logging_info(f'>> Set result folders')

    # getting device CUDA
    processing_tasks.start_task('Getting device CUDA')
    device = get_device(parameters)
    processing_tasks.finish_task('Getting device CUDA')
    
    # creating new instance of parameters file related to current running
    processing_tasks.start_task('Saving processing parameters')
    save_processing_parameters(parameters_filename, parameters)
    processing_tasks.finish_task('Saving processing parameters')

    # loading datasets and dataloaders of image dataset for processing
    processing_tasks.start_task('Loading dataloaders of image dataset')
    dataset_train, dataset_valid, dataloader_train, dataloader_valid = get_datasets_and_dataloaders(parameters)
    processing_tasks.finish_task('Loading dataloaders of image dataset')
    
    # creating neural network model 
    processing_tasks.start_task('Creating neural network model')
    model = get_neural_network_model(parameters, device)
    processing_tasks.finish_task('Creating neural network model')

    # training neural netowrk model
    processing_tasks.start_task('Training neural netowrk model')
    model = train_faster_rcnn_model(parameters, device, model, dataloader_train, dataloader_valid)
    processing_tasks.finish_task('Training neural netowrk model')

    # saving trained model and weights 
    processing_tasks.start_task('Saving trained model and weights')
    save_model_and_weights(model, parameters)
    processing_tasks.finish_task('Saving trained model and weights')
    
    # printing metrics results
    

    # finishing model training 
    logging_info('')
    logging_info('Finished the training of the model Faster RCNN' + LINE_FEED)

    # printing tasks summary 
    processing_tasks.finish_processing()
    logging_info(processing_tasks.to_string())


# ###########################################
# Methods of Level 2
# ###########################################

def get_parameters(full_path_project, parameters_filename):
    '''
    Get dictionary parameters for processing
    '''    
    # getting parameters 
    path_and_parameters_filename = os.path.join(full_path_project, parameters_filename)
    parameters = Utils.read_json_parameters(path_and_parameters_filename)


    # logging processing parameters 
    # logging_info(Utils.get_pretty_json(parameters) + LINE_FEED)   
    
    # saving current processing parameters in the log folder 
    # path_and_parameters_filename = os.path.join('log', log_filename + "-" + parameters_filename)
    # Utils.save_text_file(path_and_parameters_filename, \
    #                     Utils.get_pretty_json(parameters), 
    #                     NEW_FILE)

    # returning parameters 
    return parameters


def set_input_image_folders(parameters):
    '''
    Set folder name of input images dataset
    '''    
    
    # getting image dataset folder according processing parameters 
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    image_dataset_folder = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['input']['input_dataset']['input_dataset_path'],
        parameters['input']['input_dataset']['annotation_format'],
        input_image_size + 'x' + input_image_size,
    )

    # setting image dataset folder in processing parameters 
    parameters['processing']['image_dataset_folder'] = image_dataset_folder
    parameters['processing']['image_dataset_folder_train'] = \
        os.path.join(image_dataset_folder, 'train')
    parameters['processing']['image_dataset_folder_valid'] = \
        os.path.join(image_dataset_folder, 'valid')
    parameters['processing']['image_dataset_folder_test'] = \
        os.path.join(image_dataset_folder, 'test')


def get_running_id(parameters):
    '''
    Get last running id to calculate the current id
    '''    

    # setting control filename 
    running_control_filename = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['processing']['running_control_filename'],
    )

    # getting control info 
    running_control = Utils.read_json_parameters(running_control_filename)

    # calculating the current running id 
    running_control['last_running_id'] = int(running_control['last_running_id']) + 1

    # updating running control file 
    running_id = int(running_control['last_running_id'])

    # saving file 
    Utils.save_text_file(running_control_filename, \
                         Utils.get_pretty_json(running_control), 
                         NEW_FILE)

    # updating running id in the processing parameters 
    parameters['processing']['running_id'] = running_id

    # returning running id 
    return running_id

def set_results_folder(parameters):
    '''
    Set folder name of output results
    '''

    # creating results folders 
    main_folder = os.path.join(
        parameters['processing']['research_root_folder'],     
        parameters['training_results']['main_folder']
    )
    parameters['training_results']['main_folder'] = main_folder
    Utils.create_directory(main_folder)

    # setting and creating model folder 
    parameters['training_results']['model_folder'] = parameters['neural_network_model']['model_name']
    model_folder = os.path.join(
        main_folder,
        parameters['training_results']['model_folder']
    )
    parameters['training_results']['model_folder'] = model_folder
    Utils.create_directory(model_folder)

    # setting and creating action folder of training
    action_folder = os.path.join(
        model_folder,
        parameters['training_results']['action_folder']
    )
    parameters['training_results']['action_folder'] = action_folder
    Utils.create_directory(action_folder)

    # setting and creating running folder 
    running_id = parameters['processing']['running_id']
    running_id_text = 'running-' + f'{running_id:04}'
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    parameters['training_results']['running_folder'] = running_id_text + "-" + input_image_size + 'x' + input_image_size   
    running_folder = os.path.join(
        action_folder,
        parameters['training_results']['running_folder']
    )
    parameters['training_results']['running_folder'] = running_folder
    Utils.create_directory(running_folder)

    # setting and creating others specific folders
    processing_parameters_folder = os.path.join(
        running_folder,
        parameters['training_results']['processing_parameters_folder']
    )
    parameters['training_results']['processing_parameters_folder'] = processing_parameters_folder
    Utils.create_directory(processing_parameters_folder)

    weights_folder = os.path.join(
        running_folder,
        parameters['training_results']['weights_folder']
    )
    parameters['training_results']['weights_folder'] = weights_folder
    Utils.create_directory(weights_folder)

    # setting the base filename of weights
    weights_base_filename = parameters['neural_network_model']['model_name'] + '-' + \
                            running_id_text + "-" + input_image_size + 'x' + input_image_size
    parameters['training_results']['weights_base_filename'] = weights_base_filename
    full_model_base_filename = \
        'full_model_' + parameters['neural_network_model']['model_name'] + '-' + \
        running_id_text + "-" + input_image_size + 'x' + input_image_size
    parameters['training_results']['full_model_base_filename'] = full_model_base_filename

    metrics_folder = os.path.join(
        running_folder,
        parameters['training_results']['metrics_folder']
    )
    parameters['training_results']['metrics_folder'] = metrics_folder
    Utils.create_directory(metrics_folder)

    log_folder = os.path.join(
        running_folder,
        parameters['training_results']['log_folder']
    )
    parameters['training_results']['log_folder'] = log_folder
    Utils.create_directory(log_folder)

def get_device(parameters):
    '''
    Get device CUDA to train models
    '''    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    parameters['processing']['device'] = f'{device}'
    # print(f'Device: {device}')
    logging_info(f'Device: {device}')
    logging_info(f'')

    # print(f'torch.cuda.current_device(): {torch.cuda.current_device()}')
    # print(f'torch.cuda.get_device_name(): {torch.cuda.get_device_name()}')

    # returning current device 
    return device 

def save_processing_parameters(parameters_filename, parameters):
    '''
    Update parameters file of the processing
    '''    
    # setting full path and log folder  to write parameters file 
    path_and_parameters_filename = os.path.join(
        parameters['training_results']['processing_parameters_folder'], 
        parameters_filename)

    # saving current processing parameters in the log folder 
    Utils.save_text_file(path_and_parameters_filename, \
                        Utils.get_pretty_json(parameters), 
                        NEW_FILE)

def get_datasets_and_dataloaders(parameters):
    '''
    Get datasets and dataloaders of training, validation and testing from image dataset 
    '''

    # logging.info(f'Getting datasets')
    # logging.info(f'   Number of training images  : {len(train_dataset)}')
    # logging.info(f'   Number of validation images: {len(valid_dataset)}')
    # logging.info(f'   Total                      : {len(train_dataset) + len(valid_dataset)}')
    # logging_info(f'')

    # logging.info(f'Getting dataloaders')
    # logging.info(f'Number of training images  : {len(train_dataloader)}')
    # logging.info(f'Number of validation images: {len(valid_dataloader)}')

    # getting dataloaders from faster rcnn dataset 
    dataset_train, dataset_valid, dataloader_train, dataloader_valid = get_train_and_valid_datasets_and_dataloaders_faster_rcnn(parameters)

    # returning dataloaders from datasets for processing 
    return dataset_train, dataset_valid, dataloader_train, dataloader_valid

def get_neural_network_model(parameters, device):
    '''
    Get neural network model
    '''      
    
    # get model    
    model = get_object_detection_model(len(parameters['neural_network_model']['classes']))
     
    logging.info(f'Moving model to device: {device}')
    model = model.to(device)

    logging.info(f'Creating neural network model')
    logging.info(f'{model}')
    logging_info(f'')

    # returning neural network model
    return model

def train_faster_rcnn_model(parameters, device, model, train_dataloader, valid_dataloader):
    '''
    Execute training of the neural network model
    '''
    model = training_model(parameters, model, device, train_dataloader, valid_dataloader)

    # return trained model 
    return model 


# ###########################################
# Methods of Level 3
# ###########################################


# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
