python main.py algorithm=enkg problem=blackhole pretrain=blackhole algorithm.method.guidance_scale=4 algorithm.lr_min=1e-3

python main.py algorithm=dps_gsg problem=blackhole pretrain=blackhole algorithm.method.guidance_scale=0.01 

python main.py algorithm=scg problem=blackhole pretrain=blackhole algorithm.method.num_candidates=1024

python main.py algorithm=dpg problem=blackhole pretrain=blackhole algorithm.method.guidance_scale=10