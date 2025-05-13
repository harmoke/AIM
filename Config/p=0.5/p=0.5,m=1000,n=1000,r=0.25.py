optimizers = {
    "GD": GradientDescent(lr=1e-3),
    "HB": HeavyBall(lr=1e-4),
    "NAG": NesterovAcceleratedGradientWithLineSearch(),
    "Adagrad": Adagrad(lr=1e-1),
    "Adam": Adam(lr=8e-4, beta2=0.9999),
    "DRSOM": DRSOM(),
    "AIM_v": AIM(mtype='v'),
    "AIM_a": AIM(mtype='a'),
    "AIM_QN": AIM(mtype='QN'),
    "AIM_Hg": AIM(mtype='Hg')
}