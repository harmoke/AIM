import numpy as np
import time
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, **params):
        self.params = params
    
    @abstractmethod
    def optimize(self, problem, initial_w, **params): pass

class GradientDescent(Optimizer):
    def __init__(self, **params):
        super().__init__(**params)
        self.params.setdefault('lr', 1e-4)
        self.params.setdefault('max_iter', 10000)
        self.params.setdefault('gtol', 1e-6)
    
    def optimize(self, problem, x0):
        t_start = time.process_time()
        
        x = x0.copy()
        history = {'losses': [], 'grad_norms': []}
        grad = problem.gradient(x)
        
        for _ in range(self.params['max_iter']):
            x -= self.params['lr'] * grad
            grad = problem.gradient(x)
            
            history['losses'].append(problem.loss(x))
            history['grad_norms'].append(np.linalg.norm(grad))
            
            if history['grad_norms'][-1] < self.params['gtol']:
                break
        
        t_end = time.process_time()
        run_time = t_end - t_start
        
        return x, run_time, history
    
class HeavyBall(Optimizer):
    def __init__(self, **params):
        super().__init__(**params)
        self.params.setdefault('lr', 1e-4)
        self.params.setdefault('momentum', 0.9)
        self.params.setdefault('max_iter', 10000)
        self.params.setdefault('gtol', 1e-6)
        
    def optimize(self, problem, x0):
        t_start = time.process_time()
        
        x = x0.copy()
        history = {'trajectories': [], 'losses': [], 'grad_norms': []}
        history['trajectories'].append(x.copy())
        g = problem.gradient(x)
        x -= self.params['lr'] * g
        history['trajectories'].append(x.copy())
        
        for _ in range(self.params['max_iter']):
            g = problem.gradient(x)
            history['losses'].append(problem.loss(x))
            history['grad_norms'].append(np.linalg.norm(g))
            if history['grad_norms'][-1] < self.params['gtol']:
                break
            
            x = x - self.params['lr'] * g + self.params['momentum'] * (x - history['trajectories'][-2])
            history['trajectories'].append(x.copy())
        
        t_end = time.process_time()
        run_time = t_end - t_start
        
        return x, run_time, history
    
class NesterovAcceleratedGradientWithLineSearch(Optimizer):
    def __init__(self, **params):
        super().__init__(**params)
        self.params.setdefault('lr', 1.0)
        self.params.setdefault('beta', 0.9)
        self.params.setdefault('max_iter', 10000)
        self.params.setdefault('gtol', 1e-6)
        
    def optimize(self, problem, x0):
        t_start = time.process_time()
        
        x_k = x0.copy()
        v_k = x0.copy()
        theta_k = 1.0
        t_k = self.params['lr']
        history = {'losses': [], 'grad_norms': []}
        
        for k in range(1, self.params['max_iter'] + 1):
            y_k = (1 - theta_k) * x_k + theta_k * v_k
            
            grad_y_k = problem.gradient(y_k)
            grad_x_k = problem.gradient(x_k)
            
            history['losses'].append(problem.loss(x_k))
            grad_norm = np.linalg.norm(grad_x_k)
            history['grad_norms'].append(grad_norm)
            if grad_norm < self.params['gtol']:
                break
            
            # backtracking line search
            t_k_next = self.params['lr']
            while True:
                x_k_next = y_k - t_k_next * grad_y_k
                if problem.loss(x_k_next) <= problem.loss(y_k) - (t_k_next / 2) * np.linalg.norm(grad_y_k) ** 2:
                    break
                t_k_next *= self.params['beta']
                
            ratio = (t_k_next / t_k) * theta_k ** 2
            theta_k_next = (-np.sqrt(ratio) + np.sqrt(4 + ratio)) / 2
            
            v_k = x_k + (1 / theta_k_next) * (x_k_next - x_k)
            
            x_k = x_k_next
            theta_k = theta_k_next
            t_k = t_k_next
            
        t_end = time.process_time()
        run_time = t_end - t_start
            
        return x_k, run_time, history
    
class Adagrad(Optimizer):
    def __init__(self, **params):
        super().__init__(**params)
        self.params.setdefault('lr', 1e-4)
        self.params.setdefault('eps', 1e-8)
        self.params.setdefault('max_iter', 10000)
        self.params.setdefault('gtol', 1e-6)
        
    def optimize(self, problem, x0):
        t_start = time.process_time()
        
        x = x0.copy()
        cache = np.zeros_like(x)
        history = {'losses': [], 'grad_norms': []}
        grad = problem.gradient(x)
        
        for _ in range(self.params['max_iter']):
            cache += grad**2
            x -= self.params['lr'] * grad / np.sqrt(cache + self.params['eps'])
            grad = problem.gradient(x)
            
            history['losses'].append(problem.loss(x))
            history['grad_norms'].append(np.linalg.norm(grad))
            
            if history['grad_norms'][-1] < self.params['gtol']:
                break
        
        t_end = time.process_time()
        run_time = t_end - t_start
        
        return x, run_time, history
    
class Adam(Optimizer):
    def __init__(self, **params):
        super().__init__(**params)
        self.params.setdefault('lr', 1e-4)
        self.params.setdefault('beta1', 0.9)
        self.params.setdefault('beta2', 0.999)
        self.params.setdefault('eps', 1e-8)
        self.params.setdefault('max_iter', 10000)
        self.params.setdefault('gtol', 1e-6)
        
    def optimize(self, problem, x0):
        t_start = time.process_time()
        
        x = x0.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        history = {'losses': [], 'grad_norms': []}
        grad = problem.gradient(x)
        
        for t in range(1, self.params['max_iter']+1):
            m = self.params['beta1']*m + (1-self.params['beta1'])*grad
            v = self.params['beta2']*v + (1-self.params['beta2'])*grad**2
            
            m_hat = m / (1 - self.params['beta1']**t)
            v_hat = v / (1 - self.params['beta2']**t)
            
            x -= self.params['lr'] * m_hat / np.sqrt(v_hat + self.params['eps'])
            grad = problem.gradient(x)
            
            history['losses'].append(problem.loss(x))
            history['grad_norms'].append(np.linalg.norm(grad))
            
            if history['grad_norms'][-1] < self.params['gtol']:
                break
        
        t_end = time.process_time()
        run_time = t_end - t_start
        
        return x, run_time, history
    
class DRSOM(Optimizer):
    def __init__(self, **params):
        super().__init__(**params)
        self.params.setdefault('initial_lr', 1e-4)
        self.params.setdefault('eps', 1.0)
        self.params.setdefault('max_iter', 10000)
        self.params.setdefault('gtol', 1e-6)
        
    def optimize(self, problem, x0):
        t_start = time.process_time()
        
        x = x0.copy()
        eps = self.params['eps']
        history = {'trajectories': [], 'losses': [], 'grad_norms': []}
        
        history['trajectories'].append(x.copy())
        g = problem.gradient(x)
        x -= self.params['initial_lr'] * g
        history['trajectories'].append(x.copy())
        
        for _ in range(self.params['max_iter']):
            g_pre = g.copy()
            g = problem.gradient(x)
            history['losses'].append(problem.loss(x))
            history['grad_norms'].append(np.linalg.norm(g))
            if history['grad_norms'][-1] < self.params['gtol']:
                break
            
            Hg = (g - problem.gradient(x - eps * g)) / eps
            d = x - history['trajectories'][-2]
            Hd = g - g_pre
            
            Q = np.array([[np.inner(g, Hg), -np.inner(d, Hg)], [-np.inner(d, Hg), np.inner(d, Hd)]])
            c = np.array([np.inner(g, g), -np.inner(g, d)])
            alpha = np.linalg.inv(Q) @ c
            
            x = x - alpha[0] * g + alpha[1] * d
            history['trajectories'].append(x.copy())
            
        t_end = time.process_time()
        run_time = t_end - t_start
        
        return x, run_time, history

class AIM(Optimizer):
    def __init__(self, **params):
        super().__init__(**params)
        self.params.setdefault('initial_lr', 1e-4)
        self.params.setdefault('mtype', 'Hg')
        self.params.setdefault('beta', 1.0)
        self.params.setdefault('eta', 0.9)
        self.params.setdefault('mu', 0.75)
        self.params.setdefault('eps', 1e-3)
        self.params.setdefault('max_iter', 10000)
        self.params.setdefault('mtol', 1e-8)
        self.params.setdefault('gtol', 1e-6)
        
    def optimize(self, problem, x0):
        t_start = time.process_time()
        
        x = x0.copy()
        m = np.zeros_like(x)
        x_nxt = np.zeros_like(x)
        g_nxt = np.zeros_like(x)
        history = {'trajectories': [], 'losses': [], 'grad_norms': []}
        
        history['trajectories'].append(x.copy())
        g = problem.gradient(x)
        x -= self.params['initial_lr'] * g
        history['trajectories'].append(x.copy())
        
        beta = self.params['beta']
        mu = self.params['mu']
        eps = self.params['eps']
        eta = self.params['eta']
        
        for _ in range(self.params['max_iter']):
            g_pre = g.copy()
            g = problem.gradient(x)
            history['losses'].append(problem.loss(x))
            history['grad_norms'].append(np.linalg.norm(g))
            if history['grad_norms'][-1] < self.params['gtol']:
                break
            
            if self.params['mtype'] == 'v':
                m = x - history['trajectories'][-2]
            elif self.params['mtype'] == 'a':
                m = g - g_pre
            elif self.params['mtype'] == 'QN':
                s = x - history['trajectories'][-2]
                y = g - g_pre
                alpha = max(np.inner(s, s) / abs(np.inner(s, y)), beta / eta) * 1.1
                m = alpha * y - s
                mu = np.inner(m, m) / np.inner(m, alpha * y)
            elif self.params['mtype'] == 'Hg':
                m = (g - problem.gradient(x - eps * g)) / eps
            norm_m = np.linalg.norm(m)
            m = m / norm_m if norm_m > self.params['mtol'] else np.zeros_like(x)
            
            while True:
                x_nxt = x - beta * (g - mu * np.dot(m, g) * m)
                g_nxt = problem.gradient(x_nxt)
                
                dx = x - x_nxt
                dg = g - g_nxt
                
                r_u = np.dot(dx, dg) * beta
                r_d = np.dot(dx, dx) + mu / (1 - mu) * np.dot(m, dx)**2
                
                r = r_u / r_d
                if r > eta:
                    beta = beta * min(1.0, 1.0 / r) / 1.5
                else:
                    if r < 0.5:
                        beta = 2.0 * beta / (max(r, 0) + 1e-3)
                    break
                
            x = x_nxt.copy()
            
        t_end = time.process_time()
        run_time = t_end - t_start
            
        return x, run_time, history