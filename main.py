'''
This file runs algorithms to solve inverse problems and evaluate the results.

Inference steps:
1. instantiate the forward model
2. instantiate the dataloder for test data
3. load the pretrained diffusion model
4. run the inference algorithm

Evaluation steps:
1. instantiate the evaluation metric(s)
2. evaluate the results
'''
import os
from omegaconf import OmegaConf
import pickle
import hydra
from hydra.utils import instantiate


import torch
from torch.utils.data import DataLoader
import wandb

from utils.helper import open_url, create_logger


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.tf32:
        torch.set_float32_matmul_precision("high")
    # set random seed
    torch.manual_seed(config.seed)

    if config.wandb:
        problem_name = config.get('problem')['name']
        wandb.init(project=problem_name, group=config.algorithm.name, 
                   config=OmegaConf.to_container(config), 
                   reinit=True, settings=wandb.Settings(start_method="fork"))
        config = OmegaConf.create(dict(wandb.config)) # necessary for wandb sweep because wandb.config will be overwritten by sweep agent right after wandb.init
    # set up directory for logging and saving data
    exp_dir = os.path.join(config.problem.exp_dir, config.algorithm.name, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)
    # save config 
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))

    forward_op = instantiate(config.problem.model, device=device)
    testset = instantiate(config.problem.data)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    logger.info(f"Loaded {len(testset)} test samples...")
    # load pre-trained model
    ckpt_path = config.problem.prior

    try: 
        with open_url(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
            net = ckpt['ema'].to(device)
    except:
        net = instantiate(config.pretrain.model)
        ckpt = torch.load(config.problem.prior)
        net.load_state_dict(ckpt['ema'])
        net = net.to(device)
    
    del ckpt
    net.eval()
    if config.compile:
        net = torch.compile(net)
    logger.info(f"Loaded pre-trained model from {config.problem.prior}...")
    # set up algorithm
    algo = instantiate(config.algorithm.method, forward_op=forward_op, net=net)
    # set up evaluator

    evaluator = instantiate(config.problem.evaluator)

    for i, data in enumerate(testloader):
        data = data.to(device)
        data_id = testset.id_list[i]
        save_path = os.path.join(exp_dir, f'result_{data_id}.pt')
        if config.inference:
            # get the observation
            observation = forward_op(data)
            # run the algorithm
            logger.info(f'Running inference on test sample {data_id}...')
            results = algo.inference(observation, num_samples=config.num_samples)
            print(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')

            result_dict = {
                'observation': observation,
                'results': forward_op.unnormalize(results).cpu(),
                'data': forward_op.unnormalize(data).cpu(),
            }
            torch.save(result_dict, save_path)
            logger.info(f"Saved results to {save_path}.")
        else:
            # load the results
            result_dict = torch.load(save_path)
            logger.info(f"Loaded results from {save_path}.")
            
        # evaluate the results
        metric_dict = evaluator(pred=result_dict['results'], target=result_dict['data'], observation=result_dict['observation'])
        logger.info(f"Metric results: {metric_dict}...")
    
    logger.info("Evaluation completed...")
    # aggregate the results
    metric_state = evaluator.compute()
    logger.info(f"Final metric results: {metric_state}...")
    if config.wandb:
        wandb.log(metric_state)
        wandb.finish()


if __name__ == "__main__":
    main()