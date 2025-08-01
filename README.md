# AIM: Adaptive Inertial Method

AIM (Adaptive Inertial Method) is the numerical experiment component for the paper ["Adaptive Inertial Methods"](https://www.arxiv.org/abs/2505.15114). This repository implements and benchmarks all the adaptive inertial optimization methods discussed in the paper, providing a framework for reproducible experiments and further research.

## Project Overview

- **Purpose:** To serve as the experimental codebase for the methods and results presented in [arXiv:2505.15114](https://www.arxiv.org/abs/2505.15114).
- **Core Implementation:** All optimization algorithms described in the paper are implemented in [`optimizer.py`](./optimizer.py).
- **Reproducibility:** Scripts and configurations are provided to reproduce the numerical results and figures from the paper.

## Features

- **Comprehensive Implementations:** Adaptive inertial methods and baselines as introduced in the referenced paper, implemented in [`optimizer.py`](./optimizer.py).
- **Test Problems:** Includes logistic regression with L2 regularization and L2+Lp minimization, found in [`problems.py`](./problems.py).
- **Easy Experimentation:** Modular code for running, modifying, and extending numerical experiments.
- **Benchmarking & Visualization:** Tools for logging results and visualizing optimization performance.
- **Flexible Experimental Settings:** The `config/` folder contains files specifying parameter choices for different experiments. Each file name uniquely identifies a specific experimental setting.
- **Analysis Tools:** [`analyzer.py`](./analyzer.py) contains functions to generate tables and convergence plots for experiment results.

## Test Problems

The implemented test problems are located in [`problems.py`](./problems.py):

- **Logistic Regression with L2 Regularization:**  
  - Run [`main_logrl2.py`](./main_logrl2.py) to test logistic regression with L2 regularization.
  - You can modify parameter choices directly in `main_logrl2.py` or by pointing to different configuration files in `config/`.
- **L2 + Lp Minimization:**  
  - Run [`main_l2smoothedlp.py`](./main_l2smoothedlp.py) to test L2 + Lp minimization.
  - Parameters for this experiment can also be modified in `main_l2smoothedlp.py` or via configuration files.

## Result Analysis

- Use [`analyzer.py`](./analyzer.py) to generate:
  - Tables summarizing the results of different experimental settings.
  - Convergence graphs comparing optimization performance across methods.

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) Docker for containerized experiments

### Installation

Clone the repository:

```bash
git clone https://github.com/harmoke/AIM.git
cd AIM
```

### Usage

1. **Run experiments** using the appropriate main script:
   - For logistic regression with L2 regularization:
     ```bash
     python main_logrl2.py
     ```
   - For L2 + Lp minimization:
     ```bash
     python main_l2smoothedlp.py
     ```
   - You may modify parameters in the respective main script

## Directory Structure

```
AIM/
├── config/               # Experimental parameter choices (each file = one setting)
├── data/                 # (Optional) Datasets
├── models/               # Saved models or experiment outputs
├── scripts/              # (If present) Additional experiment runner and utilities
├── src/                  # Core adaptive inertial method implementations
│   ├── optimizer.py      # All methods from the paper
│   └── problems.py       # Test problems: logistic regression, l2+lp minimization
├── main_logrl2.py        # Run logistic regression with L2 regularization experiments
├── main_l2smoothedlp.py  # Run L2+Lp minimization experiments
├── analyzer.py           # Plot result tables and convergence graphs
├── tests/                # Unit and integration tests
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Contributing

Contributions are welcome, especially for new experiments, bug fixes, or additional visualizations.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use this code, please cite the paper:

```
@article{AIM2025,
  title={Adaptive Inertial Methods for Nonconvex Optimization},
  author={Your Authors},
  journal={arXiv preprint arXiv:2505.15114},
  year={2025}
}
```

## Contact

For questions or collaboration, please open an issue or contact [harmoke](https://github.com/harmoke).

---

*Explore and advance adaptive inertial optimization with AIM!*
