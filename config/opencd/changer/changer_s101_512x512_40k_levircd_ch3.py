_base_ = './changer_s50_512x512_40k_levircd_ch3.py'

model = dict(backbone=dict(depth=101, stem_channels=128))
