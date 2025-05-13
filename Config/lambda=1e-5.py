optimizers = {
    "GD": GradientDescent(lr=4e3),
    "HB": HeavyBall(lr=8e2),
    "NAG": NesterovAcceleratedGradientWithLineSearch(lr=1e4),
    "Adagrad": Adagrad(lr=3),
    "Adam": Adam(lr=1),
    "DRSOM": DRSOM(),
    "AIM_v": AIM(mtype='v'),
    "AIM_a": AIM(mtype='a'),
    "AIM_QN": AIM(mtype='QN'),
    "AIM_Hg": AIM(mtype='Hg')
}