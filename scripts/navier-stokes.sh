# Navier-Stokes experiments typically takes around 2-4 hours per sample (A100) as the numerical solver is much slower than evaluating diffusion model. 
python main.py pretrain=navier-stokes problem=navier-stokes problem.model.sigma_noise=0.0 algorithm=dpg exp_name=sigma0-4000 algorithm.method.guidance_scale=64.0 algorithm.method.num_mc_samples=4000 problem.model.adaptive=False
python main.py pretrain=navier-stokes problem=navier-stokes problem.model.sigma_noise=1.0 algorithm=dpg exp_name=sigma1-4000 algorithm.method.guidance_scale=64.0 algorithm.method.num_mc_samples=4000 problem.model.adaptive=False
python main.py pretrain=navier-stokes problem=navier-stokes problem.model.sigma_noise=2.0 algorithm=dpg exp_name=sigma2-4000 algorithm.method.guidance_scale=64.0 algorithm.method.num_mc_samples=4000 problem.model.adaptive=False

python main.py pretrain=navier-stokes problem=navier-stokes algorithm=dps_gsg exp_name=forward-sigma0 problem.model.adaptive=False algorithm.method.is_central=False problem.model.sigma_noise=0.0 algorithm.method.guidance_scale=0.1
python main.py pretrain=navier-stokes problem=navier-stokes algorithm=dps_gsg exp_name=forward-sigma1 problem.model.adaptive=False algorithm.method.is_central=False problem.model.sigma_noise=1.0 algorithm.method.guidance_scale=0.1
python main.py pretrain=navier-stokes problem=navier-stokes algorithm=dps_gsg exp_name=forward-sigma2 problem.model.adaptive=False algorithm.method.is_central=False problem.model.sigma_noise=2.0 algorithm.method.guidance_scale=0.1


python main.py pretrain=navier-stokes problem=navier-stokes algorithm=dps_gsg exp_name=central-sigma0 problem.model.adaptive=False algorithm.method.is_central=True problem.model.sigma_noise=0.0 algorithm.method.guidance_scale=0.1
python main.py pretrain=navier-stokes problem=navier-stokes algorithm=dps_gsg exp_name=central-sigma1 problem.model.adaptive=False algorithm.method.is_central=True problem.model.sigma_noise=1.0 algorithm.method.guidance_scale=0.1
python main.py pretrain=navier-stokes problem=navier-stokes algorithm=dps_gsg exp_name=central-sigma2 problem.model.adaptive=False algorithm.method.is_central=True problem.model.sigma_noise=2.0 algorithm.method.guidance_scale=0.1

python main.py pretrain=navier-stokes problem=navier-stokes algorithm=scg exp_name=sigma0-512 problem.model.sigma_noise=0.0 algorithm.method.num_candidates=512
python main.py pretrain=navier-stokes problem=navier-stokes algorithm=scg exp_name=sigma1-512 problem.model.sigma_noise=1.0 algorithm.method.num_candidates=512
python main.py pretrain=navier-stokes problem=navier-stokes algorithm=scg exp_name=sigma2-512 problem.model.sigma_noise=2.0 algorithm.method.num_candidates=512

python main.py pretrain=navier-stokes problem=navier-stokes algorithm=enkg exp_name=sigma0-2048 problem.model.sigma_noise=0.0 algorithm.method.guidance_scale=2.0 algorithm.method.num_samples=2048 algorithm.method.num_updates=2 algorithm.method.threshold_end=0.05 algorithm.method.threshold_start=0.05 algorithm.method.lr_min_ratio=0.0
python main.py pretrain=navier-stokes problem=navier-stokes algorithm=enkg exp_name=sigma1-2048 problem.model.sigma_noise=1.0 algorithm.method.guidance_scale=2.0 algorithm.method.num_samples=2048 algorithm.method.num_updates=2 algorithm.method.threshold_end=0.05 algorithm.method.threshold_start=0.05 algorithm.method.lr_min_ratio=0.0
python main.py pretrain=navier-stokes problem=navier-stokes algorithm=enkg exp_name=sigma2-2048 problem.model.sigma_noise=2.0 algorithm.method.guidance_scale=2.0 algorithm.method.num_samples=2048 algorithm.method.num_updates=2 algorithm.method.threshold_end=0.05 algorithm.method.threshold_start=0.05 algorithm.method.lr_min_ratio=0.0
