optimizers = {
    "GD": GradientDescent(lr=1e-3),
    "HB": HeavyBall(lr=1e-4),
    "NAG": NesterovAcceleratedGradientWithLineSearch(),
    "Adagrad": Adagrad(lr=4e-2),
    "Adam": Adam(lr=6e-3),
    "DRSOM": DRSOM(),
    "AIM_v": AIM(mtype='v'),
    "AIM_a": AIM(mtype='a'),
    "AIM_QN": AIM(mtype='QN'),
    "AIM_Hg": AIM(mtype='Hg')
}