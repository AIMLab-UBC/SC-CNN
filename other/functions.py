from __future__ import print_function, division

import torch
import time
import other.utils as utils


def train_model(model, Map, dataLoaders, criterion, val_criterion, optimizer,
                scheduler, num_epochs, d, out_size, save_name, load_flag,
                load_name):


    since = time.time()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model
    if load_flag:
        if torch.cuda.is_available():
            checkpoint = torch.load(load_name)

        else:
            checkpoint = torch.load(load_name, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss   = checkpoint['loss']

        print('Model Loaded!')

    else:
        best_loss   = 10**10
        start_epoch = 0

    best_model_wts = model.state_dict()

    for epoch in range(start_epoch, num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        val_loss   = 0.0
        train_loss = 0.0

        for phase in ['Train', 'Validation']:

            if phase == 'Train':
                # Set model to training mode
                model.train(True)

            else:
                # Set model to evaluate mode
                model.train(False)

            # Iterate over data.
            for inputs, heatmaps, epsilons in dataLoaders[phase]:

                inputs, heatmaps, epsilons = inputs.to(device), heatmaps.to(device), epsilons.to(device)

                # forward
                points, h_value = model(inputs)

                ############################    Auto Grad by me   #######################################
                predicted = Map(points, h_value, device, d, out_size)

                ############################ Auto Grad by Pytorch #######################################
                # predicted = utils.heat_map_tensor(points, h_value, device, d, out_size)
                # predicted = utils.heat_map_tensor_version2(points, h_value, device, d, out_size)

                predicted = predicted.view(predicted.shape[0], -1)
                heatmaps  = heatmaps.view(heatmaps.shape[0], -1)

                feature  = heatmaps.shape[1]
                epsilons = epsilons.unsqueeze(1)
                epsilons = torch.repeat_interleave(epsilons, repeats=feature, dim=1)

                weights = heatmaps + epsilons

                if phase == 'Train':

                    ###############################################
                    # BCE
                    # loss = criterion(predicted, heatmaps)
                    # weight_loss = loss * weights

                    ###############################################
                    # SC_CNN
                    weight_loss = criterion(predicted, heatmaps, weights)

                    # Sum over one data
                    # Average over different data
                    sum_loss = torch.sum(weight_loss, dim=1)
                    # avg_loss = torch.mean(sum_loss)
                    avg_loss = torch.sum(sum_loss)

                    # Just for having understanding what is happening
                    train_loss += avg_loss.item()

                else:

                    loss = val_criterion(predicted, heatmaps)
                    RMSE = torch.sqrt(loss)
                    RMSE = torch.sum(RMSE, dim=1)
                    # RMSE = torch.mean(RMSE)
                    RMSE = torch.sum(RMSE)

                    val_loss += RMSE.item()


                if phase == 'Train':

                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Calculate gradient respect to parameters
                    avg_loss.backward()
                    # Update parameters
                    optimizer.step()
                    scheduler.step()

                # Empty Catch
                torch.cuda.empty_cache()

        print()
        if val_loss < best_loss:

            best_loss = val_loss
            best_model_wts = model.state_dict()

            # IF using Multiple GPUs, the model should be saved in this way
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_name)
            else:
                torch.save(model.state_dict(), save_name)

            torch.save({'epoch'  : epoch,
                        'model_state_dict'    : model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss'                : val_loss
                        }, save_name)

            print('**********************************')
            print('* Finding Better Model --> Save  *')
            print('**********************************')
            print()


        print('Training Loss is:', train_loss, ' and Validation Loss is:', val_loss)
        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
           time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model
