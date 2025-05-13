import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from problems import LogisticRegressionL2
from optimizers import *
from analyzer import ResultAnalyzer

def main():
    A, b = load_svmlight_file('Data/real-sim')
    b = (b == 1).astype(int)
    x0 = np.zeros(A.shape[1])
    
    params = {
        "A": A,
        "b": b,
        "l": 1e-7
    }
    
    problem = LogisticRegressionL2(**params)
    
    analyzer = ResultAnalyzer(problem, x0)
    
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
    
    for name, optimizer in optimizers.items():
        x, run_time, history = optimizer.optimize(problem, x0.copy())
        analyzer.add_result(name, run_time, history)
    
    print("=== 结果对比 ===")
    analyzer.print_table()
    print("\n=== 收敛曲线 ===")
    analyzer.plot_convergence()

if __name__ == "__main__":
    main()