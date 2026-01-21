# Efficient and Adaptive Pong Control via Gated Cell Ensembles

This project was created during the Machine Learning course at PWR under supervision of Weronika Węgier. The final project report can be found in [docs/project_report.pdf](docs/project_report.pdf).

## Research Summary

### Objective
Develop a lightweight, online-adaptive classifier for real-time Pong control under non-stationary ball dynamics, comparing the proposed **Gated Cell Ensemble (GCE)** against Decision Tree and Hoeffding Tree baselines.

### Methodology
- **Environment**: Custom Pong (900x600px, 60 FPS, ball acceleration 1.1x per hit)
- **Models**: Decision Tree (sklearn), Hoeffding Tree (River), Gated Cell Ensemble (custom MoE-style)
- **Training**: Offline pretraining on gameplay data + online self-supervised learning via trajectory prediction oracle
- **Offline Validation**: Repeated Stratified K-Fold (5 splits × 10 repeats)
- **Online Evaluation**: 20 games per model against PC opponent (first to 5 points)
- **Statistical Analysis**: Friedman test, Wilcoxon signed-rank test with Hommel correction

### Key Results

**Offline Cross-Validation:**

| Model | Accuracy | Latency |
|-------|----------|---------|
| Decision Tree | **88.12 ± 0.16%** | **0.10 μs** |
| Hoeffding Tree | 63.08 ± 0.34% | 10.10 μs |
| Gated Cell Ensemble | 43.54 ± 0.45% | 102.88 μs |

**Online Simulation:**

| Model | Variant | Survival Time | Avg. Returns |
|-------|---------|---------------|--------------|
| Decision Tree | Static | 30.37 ± 7.92s | 4.55 ± 2.33 |
| Hoeffding Tree | Static | 30.65 ± 7.19s | 4.75 ± 2.07 |
| Hoeffding Tree | Adaptive | **35.00 ± 10.56s** | **6.30 ± 3.67** |
| Gated Cell Ensemble | Static | 23.18 ± 9.95s | 2.45 ± 2.65 |
| Gated Cell Ensemble | Adaptive | 14.09 ± 0.85s | 0.05 ± 0.22 |

### Conclusion
**The Adaptive Hoeffding Tree proved most robust for online adaptation**, significantly outperforming static baselines. The proposed GCE architecture showed promise in offline partitioning but suffered from instability during online learning due to vector oscillation from the boolean reward signal. The standard Decision Tree remained competitive as a static baseline with highest offline accuracy. Future work should address GCE stability through alternative update mechanisms.

## Development

### Setup

```bash
pip install -r requirements.txt
```

### Notebook
To replicate experiments and results, use `notebook.ipynb` which contains the complete workflow: data setup, model training, offline/online evaluation, and statistical analysis.

### Running the Game

```bash
python src/main.py --mode {human|pc|dt|ht|wf}
```
