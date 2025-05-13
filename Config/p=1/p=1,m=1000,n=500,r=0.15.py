optimizers = {
    "GD": GradientDescent(lr=3e-3),
    "HB": HeavyBall(lr=3e-4),
    "NAG": NesterovAcceleratedGradientWithLineSearch(),
    "Adagrad": Adagrad(lr=4e-2),
    "Adam": Adam(lr=9e-3),
    "DRSOM": DRSOM(),
    "AIM_v": AIM(mtype='v'),
    "AIM_a": AIM(mtype='a'),
    "AIM_QN": AIM(mtype='QN'),
    "AIM_Hg": AIM(mtype='Hg')
}