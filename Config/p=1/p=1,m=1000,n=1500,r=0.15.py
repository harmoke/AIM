optimizers = {
    "GD": GradientDescent(lr=2e-3),
    "HB": HeavyBall(lr=2e-4),
    "NAG": NesterovAcceleratedGradientWithLineSearch(),
    "Adagrad": Adagrad(lr=4.2e-2),
    "Adam": Adam(lr=4e-3, beta2=0.9999),
    "DRSOM": DRSOM(),
    "AIM_v": AIM(mtype='v'),
    "AIM_a": AIM(mtype='a'),
    "AIM_QN": AIM(mtype='QN'),
    "AIM_Hg": AIM(mtype='Hg')
}