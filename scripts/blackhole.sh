python main.py algorithm=dps_gsg problem=blackhole pretrain=blackhole algorithm.method.guidance_scale=0.01 

python main.py algorithm=scg problem=blackhole pretrain=blackhole algorithm.method.num_candidates=1024

python main.py algorithm=dpg problem=blackhole pretrain=blackhole algorithm.method.guidance_scale=10

python main.py pretrain=blackhole problem=blackhole algorithm=enkg algorithm.method.guidance_scale=3.5 algorithm.method.num_samples=1024 algorithm.method.num_updates=1 algorithm.method.threshold_end=0.1 algorithm.method.threshold_start=0.1 algorithm.method.lr_min_ratio=0.001

