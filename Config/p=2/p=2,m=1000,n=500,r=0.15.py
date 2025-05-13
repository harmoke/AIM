optimizers = {
    "GD": GradientDescent(lr=4e-3),
    "HB": HeavyBall(lr=4e-4),
    "NAG": NesterovAcceleratedGradientWithLineSearch(),
    "Adagrad": Adagrad(lr=10),
    "Adam": Adam(lr=3e-3),
    "DRSOM": DRSOM(),
    "AIM_v": AIM(mtype='v'),
    "AIM_a": AIM(mtype='a'),
    "AIM_QN": AIM(mtype='QN'),
    "AIM_Hg": AIM(mtype='Hg')
}