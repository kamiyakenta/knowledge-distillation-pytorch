"""
Main entrance for train/eval with/without KD on CIFAR-10
e.g.
python3 cloud_ai_main.py --model_dir experiments/cloud_ai_*
"""
import argparse
import logging
import os
import random
import time
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import utils
from evaluate import evaluate
from model import data_loader, net, resnet, wrn
parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--cpu', action='store_true')
def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """
    # set model to training mode
    model.train()
    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), \
                                            labels_batch.cuda(async=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            # performs updates using calculated gradients
            optimizer.step()
            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()
                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)
            # update the average loss
            loss_avg.update(loss.data[0])
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    best_val_acc = 0.0
    # learning rate schedulers for different models:
    if params.model_version == "resnet18":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    for epoch in range(params.num_epochs):
        scheduler.step()
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)
        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)        
        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
if __name__ == '__main__':
    start = time.time()
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'cloud_ai_params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # use GPU if available
    params.cuda = torch.cuda.is_available()
    if args.cpu:
        params.cuda = False
    # Set the random seed for reproducible experiments
    random.seed(230)
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("Loading the datasets...")
    # fetch dataloaders, considering full-set vs. sub-set scenarios
    if params.subset_percent < 1.0:
        train_dl = data_loader.fetch_subset_dataloader('train', params)
    else:
        train_dl = data_loader.fetch_dataloader('train', params)
    dev_dl = data_loader.fetch_dataloader('dev', params)
    logging.info("- done.")
    if params.model_version == "cnn":
        model = net.Net(params).cuda() if params.cuda else net.Net(params)
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        loss_fn = net.loss_fn
        metrics = net.metrics
    elif params.model_version == "resnet18":
        model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                momentum=0.9, weight_decay=5e-4)
        loss_fn = resnet.loss_fn
        metrics = resnet.metrics
    elif params.model_version == "wrn":
        model = wrn.wrn(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
        model = model.cuda() if params.cuda else model
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                momentum=0.9, weight_decay=5e-4)
        loss_fn = wrn.loss_fn
        metrics = wrn.metrics
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, params,
                       args.model_dir, args.restore_file)
    end_time = time.time() - start
    logging.info("~~~~~~~~~全体時間~~~~~~~~~")
    logging.info(str(end_time) + "[sec]")
