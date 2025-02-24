python main.py pretrain=navier-stokes problem=navier-stokes_ds2 problem.model.sigma_noise=0.0 algorithm=dpg exp_name=sigma0-4000 algorithm.method.guidance_scale=64.0 algorithm.method.num_mc_samples=4000 problem.model.adaptive=False
python main.py pretrain=navier-stokes problem=navier-stokes_ds2 problem.model.sigma_noise=1.0 algorithm=dpg exp_name=sigma1-4000 algorithm.method.guidance_scale=64.0 algorithm.method.num_mc_samples=4000 problem.model.adaptive=False
python main.py pretrain=navier-stokes problem=navier-stokes_ds4 problem.model.sigma_noise=2.0 algorithm=dpg exp_name=sigma2-4000 algorithm.method.guidance_scale=64.0 algorithm.method.num_mc_samples=4000 problem.model.adaptive=False

python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg exp_name=forward-sigma0 problem.model.adaptive=False algorithm.method.is_central=False problem.model.sigma_noise=0.0 algorithm.method.guidance_scale=0.1
python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg exp_name=forward-sigma1 problem.model.adaptive=False algorithm.method.is_central=False problem.model.sigma_noise=1.0 algorithm.method.guidance_scale=0.1
python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg exp_name=forward-sigma2 problem.model.adaptive=False algorithm.method.is_central=False problem.model.sigma_noise=2.0 algorithm.method.guidance_scale=0.1


python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg exp_name=central-sigma0 problem.model.adaptive=False algorithm.method.is_central=True problem.model.sigma_noise=0.0 algorithm.method.guidance_scale=0.1
python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg exp_name=central-sigma1 problem.model.adaptive=False algorithm.method.is_central=True problem.model.sigma_noise=1.0 algorithm.method.guidance_scale=0.1
python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=dps_gsg exp_name=central-sigma2 problem.model.adaptive=False algorithm.method.is_central=True problem.model.sigma_noise=2.0 algorithm.method.guidance_scale=0.1

python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=scg exp_name=sigma0-512 problem.model.sigma_noise=0.0 algorithm.method.num_candidates=512
python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=scg exp_name=sigma1-512 problem.model.sigma_noise=1.0 algorithm.method.num_candidates=512
python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=scg exp_name=sigma2-512 problem.model.sigma_noise=2.0 algorithm.method.num_candidates=512

python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=enkg exp_name=sigma0-2048 problem.model.sigma_noise=0.0 algorithm.method.guidance_scale=2.0 algorithm.method.num_samples=2048 algorithm.method.num_updates=2
python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=enkg exp_name=sigma1-2048 problem.model.sigma_noise=1.0 algorithm.method.guidance_scale=2.0 algorithm.method.num_samples=2048 algorithm.method.num_updates=2
python main.py pretrain=navier-stokes problem=navier-stokes_ds2 algorithm=enkg exp_name=sigma2-2048 problem.model.sigma_noise=2.0 algorithm.method.guidance_scale=2.0 algorithm.method.num_samples=2048 algorithm.method.num_updates=2
