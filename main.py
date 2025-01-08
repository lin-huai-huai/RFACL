import torch
import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader_GNN import data_generator

from trainer.trainer_GNN import Trainer, model_evaluate
from models.TC_GNN import TC

from utils import _calc_metrics, copy_Files
from models.model_GNN import Base_model

start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='RFACL', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='train_linear', type=str)
parser.add_argument('--selected_dataset', default='ArticularyWordRecognition', type=str)
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

def main(configs, args,lambda1, lambda2, lambda3,results1, results_F1):
    device = torch.device(args.device)
    experiment_description = args.experiment_description

    method = 'RFACL'
    training_mode = args.training_mode
    run_description = args.run_description

    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok=True)

    # ##### fix random seeds for reproducibility ########
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    #####################################################

    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    # Logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {method}')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug("=" * 45)

    data_path = f"./data/{data_type}"
    train_dl, test_dl = data_generator(data_path, configs, args)
    logger.debug("Data loaded ...")

    # Load Model
    model = Base_model(configs, args).to(device)
    temporal_contr_model = TC(configs, args, device).to(device)

    if training_mode == "train_linear" or "tl" in training_mode:
        load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        del_list = ['logits']
        pretrained_dict_copy = pretrained_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del pretrained_dict[i]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        set_requires_grad(model, pretrained_dict, requires_grad=False)


    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

    if training_mode == "self_supervised":
        copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

    # Trainer
    Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode,
            lambda1, lambda2, lambda3,results1, results_F1)

    if training_mode != "self_supervised":
        # Testing
        outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        total_loss, total_acc, valid_f1, pred_labels, true_labels = outs
        _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

    logger.debug(f"Training time is : {datetime.now()-start_time}")

if __name__ == '__main__':

    dataset_UEA = ['ArticularyWordRecognition']
    default_lambda1 = 1
    default_lambda2 = 0.7
    default_lambda3 = 0.7
    args = parser.parse_args()

    args.selected_dataset = 'ArticularyWordRecognition'
    data_type = args.selected_dataset
    exec(f'from config_files.{data_type}_Configs import Config as Configs')

    configs = Configs()

    results1 = []
    results_F1 = []
    for j in range(10):

        args.training_mode = 'self_supervised'
        main(configs, args, default_lambda1, default_lambda2, default_lambda3,results1, results_F1)
        args.training_mode = 'train_linear'
        main(configs, args, default_lambda1, default_lambda2, default_lambda3,results1, results_F1)

    total_acc = sum(results1)
    print(total_acc)
    total_F1 = sum(results_F1)
    print(total_F1)

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{curr_time}.txt"
    with open(filename, 'w') as file:
        file.write(f"Total Acc: {total_acc}\n")
        file.write(f"Total F1: {total_F1}\n")
        file.write("Results List:\n")
        for result in results1:
            file.write(f"{result}\n")
        file.write("F1 List:\n")
        for result in results_F1:
            file.write(f"{result}\n")

    print(f"Results saved to {filename}")
