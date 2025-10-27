# 🧠 ReML — Rebuilding Machine Learning from Scratch

**ReML** (Rebuild Machine Learning) is a personal research and learning framework designed to reimplement the core concepts of modern ML libraries — such as **NumPy**, **SciPy**, **Scikit-learn**, and **PyTorch** — entirely from scratch.

The goal is to deeply understand how these libraries work internally by rebuilding their components step by step, testing them against the real implementations, and documenting the learning process.

---

## 🚀 Project Goals

- Understand **mathematical foundations** behind ML algorithms  
- Implement **core functions** (matrix operations, optimization, scaling, etc.)  
- Build **modular ML algorithms** using real-world library structures (`sklearn`, `scipy`, `torch`)  
- Compare and **benchmark results** with standard libraries  
- Create a **learning and research playground** for experimentation  

## 📁 Folder Structure
```bash
ReML/
├── reml/                      # Core package (custom & from-scratch implementations)
│
├── experiments/               # Scripts and notebooks for exploratory analysis and model comparison
├── tests/                     # Unit and integration tests for all core modules
├── benchmarks/                # Performance evaluations vs standard libraries (NumPy, SciPy, scikit-learn, PyTorch)
├── data/                      # Sample datasets for testing and experimentation
├── notebooks/                 # Concept derivations, algorithm walkthroughs, and visualizations
├── reports/                   # Notes, summaries, and experimental results
└── README.md                  # Project overview and documentation

```

## 🧠 Learning Workflow

- Implement algorithms from scratch inside reml/
- Experiment with notebooks in experiments/
- Compare your results with standard libraries
- Test using pytest in tests/
- Visualize results in notebooks/
- Document your learnings in reports/

## 🔧 Requirements

- Python ≥ 3.10
- NumPy (for initial testing and comparison)
- Matplotlib (optional, for visualization)
- Pytest (for testing)

## 📘 Status

| Module         | Status         | Description                                  |
| -------------- | -------------- | ---------------------------------------------|
| `reml.numpy`   | 🟢 In Progress | Basic array and linear algebra operations    |
| `reml.scipy`   | 🔵 Planned     | Optimization and numerical routines          |
| `reml.sklearn` | 🟢 In Progress | Core ML algorithms (Linear Regression, etc.) |
| `reml.torch`   | 🔵 Planned     | Neural network components                    |

## 📚 Future Plans

- Build neural network layer system

## 🧑‍💻 Author

**Dheeraj Karki**
Passionate about AI application, and interdisciplinary research.
Exploring how to build intelligence from fundamentals — one algorithm at a time.

> “The best way to understand something deeply is to build it from scratch.” 🧩