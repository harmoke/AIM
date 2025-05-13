import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tabulate import tabulate

class BenchmarkSolver:
    @staticmethod
    def find_optimal(problem, x0):
        res = minimize(
            fun=problem.loss,
            x0=x0,
            jac=problem.gradient,
            method='L-BFGS-B',
            options={
                'disp': 1,
                'ftol': 1e-16,
                'gtol': 1e-8,
                'maxiter': 10000
            }
        )
        if not res.success:
            raise RuntimeError(f"Baseline solver failures: {res.message}")
        return res.x, res.fun

class ResultAnalyzer:
    def __init__(self, problem, x0):
        self.problem = problem
        self.x0 = x0
        self.results = []
        self.x_star, self.f_star = self._calculate_optimal()
    
    def _calculate_optimal(self):
        return BenchmarkSolver.find_optimal(self.problem, self.x0)
    
    def add_result(self, name, run_time, history):
        self.results.append({
            'name': name,
            'iterations': len(history['losses']),
            'run_time': run_time,
            'final_grad_norm': history['grad_norms'][-1],
            'final_loss': history['losses'][-1],
            'history': history
        })
    
    def print_table(self):
        table = []
        for res in self.results:
            table.append([
                res['name'],
                res['iterations'],
                f"{res['run_time']:.6f}",
                f"{res['final_grad_norm']:.6e}",
                f"{res['final_loss']:.6e}",
                f"{res['final_loss'] - self.f_star:.6e}"
            ])
        print(tabulate(table, 
            headers=["Algorithm", "Iterations", "CPU time", "Grad Norm", "Loss", "Optimality Gap"],
            tablefmt="github"))
    
    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        
        # 动态生成颜色，最多10个
        num_results = min(len(self.results), 10)  # 限制最大10条线
        
            # 定义8种低调但有区分度的颜色
        base_color_pool = [
            '#1E90FF',  # 道奇蓝
            '#2F4F4F',  # 暗板岩灰
            '#9932CC',  # 暗兰花紫
            '#8B4513',  # 鞍褐
            '#228B22',  # 森林绿
            '#D2691E',  # 巧克力色
            '#483D8B',  # 暗石板蓝
            '#CD5C5C',  # 印度红
        ]
    
        # 最后两个显眼的颜色
        highlight_colors = ['#FFA500', '#FF0000']  # 橙色和红色
    
        # 根据结果数量选择颜色
        if num_results <= 2:
            colors = highlight_colors[:num_results]  # 1或2个结果时只用显眼色
        else:
            num_base = min(num_results - 2, 8)  # 最多8个低调色
            base_colors = base_color_pool[:num_base]
            colors = base_colors + highlight_colors[:(num_results - num_base)]
            
        # 绘制每条曲线
        for idx, res in enumerate(self.results[:10]):  # 限制最多10条
            gaps = np.array(res['history']['losses']) - self.f_star
            iterations = np.arange(1, len(gaps)+1)
    
            plt.semilogy(iterations, gaps, label=res['name'], color=colors[idx])
            
#       plt.title("Convergence Analysis", fontsize=14)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel(r"$f(x) - f^*$", fontsize=12)
        
        max_iter = max(len(res['history']['losses']) for res in self.results)
        plt.xticks(
            ticks=np.arange(1, max_iter+1, step=max(1, max_iter//10)),
            labels=np.arange(1, max_iter+1, step=max(1, max_iter//10)).astype(int)
        )
        
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        plt.xlim(left=0.5, right=max_iter+0.5)
        
        plt.tight_layout()
        plt.show()