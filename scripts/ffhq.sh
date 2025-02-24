python main.py problem=ffhq256_inpaint pretrain=ffhq256 algorithm=dpg 
python main.py problem=ffhq256_sr pretrain=ffhq256 algorithm=dpg algorithm.method.guidance_scale=160.0 algorithm.method.num_mc_samples=800 algorithm.method.batch_size=200
python main.py problem=ffhq256_deblur pretrain=ffhq256 algorithm=dpg algorithm.method.guidance_scale=200.0 algorithm.method.num_mc_samples=800 algorithm.method.batch_size=200
python main.py problem=ffhq256_pr pretrain=ffhq256 algorithm=dpg algorithm.method.guidance_scale=300.0 algorithm.method.num_mc_samples=1000 algorithm.method.batch_size=500

python main.py problem=ffhq256_inpaint pretrain=ffhq256 algorithm=scg algorithm.method.num_candidates=256
python main.py problem=ffhq256_sr pretrain=ffhq256 algorithm=scg algorithm.method.num_candidates=256
python main.py problem=ffhq256_deblur pretrain=ffhq256 algorithm=scg algorithm.method.num_candidates=256
python main.py problem=ffhq256_pr pretrain=ffhq256 algorithm=scg algorithm.method.num_candidates=256


python main.py pretrain=ffhq256 problem=ffhq256_sr algorithm=dps_gsg exp_name=forward-sigma0 algorithm.method.is_central=False algorithm.method.guidance_scale=1.0
python main.py pretrain=ffhq256 problem=ffhq256_inpaint algorithm=dps_gsg exp_name=forward-sigma1 algorithm.method.is_central=False algorithm.method.guidance_scale=1.0
python main.py pretrain=ffhq256 problem=ffhq256_deblur algorithm=dps_gsg exp_name=forward-sigma2 algorithm.method.is_central=False algorithm.method.guidance_scale=1.0
python main.py pretrain=ffhq256 problem=ffhq256_pr algorithm=dps_gsg exp_name=forward-sigma2 algorithm.method.is_central=False algorithm.method.guidance_scale=1.0


python main.py pretrain=ffhq256 problem=ffhq256_sr algorithm=dps_gsg exp_name=central-sigma0 algorithm.method.is_central=True algorithm.method.guidance_scale=1.0
python main.py pretrain=ffhq256 problem=ffhq256_inpaint algorithm=dps_gsg exp_name=central-sigma1 algorithm.method.is_central=True algorithm.method.guidance_scale=1.0
python main.py pretrain=ffhq256 problem=ffhq256_deblur algorithm=dps_gsg exp_name=central-sigma2 algorithm.method.is_central=True algorithm.method.guidance_scale=1.0
python main.py pretrain=ffhq256 problem=ffhq256_pr algorithm=dps_gsg exp_name=forward-sigma2 algorithm.method.is_central=True algorithm.method.guidance_scale=1.0

python main.py pretrain=ffhq256 problem=ffhq256_sr algorithm=enkg algorithm.method.guidance_scale=2.0 algorithm.method.num_samples=1024 algorithm.method.num_updates=2
python main.py pretrain=ffhq256 problem=ffhq256_inpaint algorithm=enkg algorithm.method.guidance_scale=2.0 algorithm.method.num_samples=1024 algorithm.method.num_updates=2
python main.py pretrain=ffhq256 problem=ffhq256_deblur algorithm=enkg algorithm.method.guidance_scale=2.0 algorithm.method.num_samples=1024 algorithm.method.num_updates=2
python main.py pretrain=ffhq256 problem=ffhq256_pr algorithm=enkg algorithm.method.guidance_scale=2.0 algorithm.method.num_samples=1024 algorithm.method.num_updates=2
