optimizers = {
    "GD": GradientDescent(lr=7e3),
    "HB": HeavyBall(lr=7e3),
    "NAG": NesterovAcceleratedGradientWithLineSearch(lr=4e4),
    "Adagrad": Adagrad(lr=10),
    "Adam": Adam(lr=10),
    "DRSOM": DRSOM(),
    "AIM_v": AIM(mtype='v'),
    "AIM_a": AIM(mtype='a'),
    "AIM_QN": AIM(mtype='QN'),
    "AIM_Hg": AIM(mtype='Hg')
}