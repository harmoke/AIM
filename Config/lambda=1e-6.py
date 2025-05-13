optimizers = {
    "GD": GradientDescent(lr=7e3),
    "HB": HeavyBall(lr=1.4e3),
    "NAG": NesterovAcceleratedGradientWithLineSearch(lr=2e4),
    "Adagrad": Adagrad(lr=5),
    "Adam": Adam(lr=4),
    "DRSOM": DRSOM(),
    "AIM_v": AIM(mtype='v'),
    "AIM_a": AIM(mtype='a'),
    "AIM_QN": AIM(mtype='QN'),
    "AIM_Hg": AIM(mtype='Hg')
}