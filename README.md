
# PoseCompare

**Visual Comparison Tool for Popular Pose Detection Models**

PoseCompare is a tool designed to visually compare the performance of popular pose detection models. It enables users to analyze pose landmarks, joint positions, and motion outputs to evaluate and contrast model accuracy, smoothness, and responsiveness.

---

## 📖 Overview

PoseCompare provides:
- Side-by-side visual comparisons of pose detection outputs.
- Landmark visualization and extracted motion rules.
- Model evaluation for fitness tracking, animation, and other computer vision applications.

**Supported Pose Detection Models:**
- ViTPose
- mediapipe
- 4DHumans


---

## 🚀 How to Install and Run

### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **Conda** (Anaconda or Miniconda)
- **Git** (to clone the repository)
- Internet access (for downloading model weights).

### Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/posecompare.git
   cd posecompare
   ```

2. **Set up the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate posecompare_env
   ```

3. **Download model weights:**
   ```bash
   curl -o weights.pth https://your-model-weights-url.com/weights.pth
   ```

4. **Run the tool:**
   - Open the notebook interface:
     ```bash
     jupyter notebook PoseCompare.ipynb
     ```
   - Or execute the notebook programmatically:
     ```bash
     jupyter nbconvert --to notebook --execute PoseCompare.ipynb --output executed_PoseCompare.ipynb
     ```

---

## 🛠 Development Plan

### Current Features
- [x] Environment setup using `environment.yml`.
- [x] Visualize pose landmarks and joint positions.
- [x] Support multiple pose detection models for comparison.

### Planned Features
- [ ] Implement smoothing algorithms for motion data.
- [ ] Enhance GUI for improved usability.
- [ ] Add benchmarking tools for model performance metrics.
- [ ] Allow user-defined pose detection model integration.
- [ ] Provide export options for analysis results and visualizations.

---

## 🤝 Contributing

We welcome contributions, feedback, and feature requests!  
Feel free to:
- Open an issue to report bugs or suggest improvements.
- Submit a pull request to contribute code or documentation updates.

---

## 📄 License

PoseCompare is open-source and distributed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
