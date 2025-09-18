# Laplace Equation Visualizer

This is an interactive Streamlit web app for solving and visualizing the **2D Laplace equation** with customizable boundary conditions.  

This project is designed for physics and engineering students to explore how electrostatic potentials and fields arise, while also learning about the computational methods behind numerical solutions.  

You can either run it locally or try it via URL in your browser — no installation needed!  

---

## About

This project helps students develop intuition for how **electrostatic potentials and fields** emerge from solving **Laplace’s equation**.  

The app shows how boundary conditions shape the potential and field lines, making it easier to connect theory with visualization, building intuition. At the same time, it keeps a strong focus on **computational physics concepts**:  

- Gauss–Seidel iteration with Successive Over-Relaxation (SOR)  
- Effects of grid resolution and discretization error  
- How tolerance choices influence convergence  
- Connections between numerical methods and physical interpretation  

In other words, the app is both a **learning tool for physics intuition** and a **gentle introduction to computational methods** commonly used in scientific computing. 

---

## Features

- Adjustable grid size and boundary voltages  
- Gauss–Seidel solver with SOR (animated convergence optional)  
- 2D potential maps with selectable colormaps  
- Electric field visualization via finite differences and streamlines  
- 1D cross-sections of potential  
- 3D surface plots (static with matplotlib and interactive with Plotly)  
- Solver feedback with recommended ω and minimum meaningful tolerance  

---

## Usage

### 1. Run online (no installation needed)  
Try the app instantly in your browser: [Laplace Visualizer Web App](https://laplace-visualizer.streamlit.app/)  

### 2. Run locally  

```bash
git clone https://github.com/lucy-cook/Laplace-Visualizer
cd laplace-visualizer
pip install -r requirements.txt
streamlit run laplace2.py
```

---
## How it Works
- Initializes a potential grid with Dirichlet boundary conditions from user input.
- Uses Gauss–Seidel iteration with SOR to update interior points until convergence.
- Computes the electric field via finite-difference gradients.
- Provides multiple visualization options: 2D heatmaps, streamlines, 1D slices, and 3D surfaces.

---
Contributions welcome! Thanks for reading!

<3 Lucy 
