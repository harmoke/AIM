optimizers = {
    "GD": GradientDescent(lr=2e-3),
    "HB": HeavyBall(lr=2e-4),
    "NAG": NesterovAcceleratedGradientWithLineSearch(),
    "Adagrad": Adagrad(lr=3e-1),
    "Adam": Adam(lr=1e-2),
    "DRSOM": DRSOM(),
    "AIM_v": AIM(mtype='v'),
    "AIM_a": AIM(mtype='a'),
    "AIM_QN": AIM(mtype='QN'),
    "AIM_Hg": AIM(mtype='Hg')
}