import numpy as np
from problems import SmoothedLpL2Problem
from optimizers import *
from analyzer import ResultAnalyzer

def generate_data(m, n, sparsity=0.15, random_seed=42):
    """
    Generate sparse regression test data
        Parameters:
            m : int 
                Number of samples (matrix rows)
            n : int 
                Number of features (matrix columns)
            sparsity : float (default=0.15)
                Sparsity ratio of matrix A
            random_seed : int (default=42)
                Random seed
        
        Returns:
            A : ndarray (m, n)  Sparse design matrix
            b : ndarray (m,)    Observation vector
    """
    rng = np.random.default_rng(random_seed)
    
    total_elements = m * n
    non_zero_num = int(total_elements * sparsity)
    flat_indices = rng.choice(total_elements, non_zero_num, replace=False)
    rows, cols = np.unravel_index(flat_indices, (m, n))
    A = np.zeros((m, n))
    A[rows, cols] = rng.normal(loc=0.0, scale=1.0, size=non_zero_num)
    
    mask = rng.binomial(1, 0.5, size=n).astype(bool)
    v = np.zeros(n)
    v[~mask] = rng.normal(loc=0.0, scale=np.sqrt(1/n), size=np.sum(~mask))
    
    delta = rng.normal(loc=0.0, scale=1.0, size=m)
    
    b = A @ v + delta
    
    return A, b

def main():
    A, b = generate_data(1000, 1500, 0.25)
    x0 = np.zeros(A.shape[1])
    
    params = {
        "A": A,
        "b": b,
        "p": 2
    }
    
    problem = SmoothedLpL2Problem(**params)
    analyzer = ResultAnalyzer(problem, x0)
    
    optimizers = {
        "GD": GradientDescent(lr=1e-3),
        "HB": HeavyBall(lr=1e-4),
        "NAG": NesterovAcceleratedGradientWithLineSearch(),
        "Adagrad": Adagrad(lr=1),
        "Adam": Adam(lr=1e-2),
        "DRSOM": DRSOM(),
        "AIM_v": AIM(mtype='v'),
        "AIM_a": AIM(mtype='a'),
        "AIM_QN": AIM(mtype='QN'),
        "AIM_Hg": AIM(mtype='Hg')
    }
    
    for name, optimizer in optimizers.items():
        x, run_time, history = optimizer.optimize(problem, x0.copy())
        analyzer.add_result(name, run_time, history)
    
    print("=== Result Table ===")
    analyzer.print_table()
    print("\n=== Convergence Plot ===")
    analyzer.plot_convergence()

if __name__ == "__main__":
    main()