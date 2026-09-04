# ug-diffusion-bbn

Pipeline en Python para evaluar Gravedad Unimodular con término de difusión en BBN. Implementa solvers físicos de abundancias de elementos livianos, inferencia estadística bayesiana (MCMC) y un emulador con redes neuronales en PyTorch para aceleración.

---

## **Descripción General**

Este repositorio estudia la compatibilidad de la **Gravedad Unimodular (UG)** con un término de difusión $Q(T)$ frente a las restricciones observacionales de la **Nucleosíntesis Primordial (BBN)**.

En UG, la no conservación del tensor energía-impulso altera la tasa de expansión de Friedmann:
$$3 H^2 M_P^2 = \rho_{\text{rad}}(T) + Q(T)$$

Para evaluar el impacto físico sobre la abundancia de helio-4 ($Y_p$) y deuterio ($D/H$), el código mapea la densidad de energía adicional de $Q(T)$ en el desacoplamiento débil ($T \sim 0.7 \text{ MeV}$) hacia un número efectivo de neutrinos equivalente $\Delta N_{\text{eff}}$. La respuesta de BBN se evalúa interpolando de forma precisa las tablas numéricas de **PRIMAT / PArthENoPE** a través de `camb.bbn`.

---

## **Estructura del Proyecto**

```text
ug-diffusion-bbn/
├── src/
│   ├── unimodular_physics.py  # Módulo físico: Q(T), rho_rad(T) y mapa Delta_Neff
│   ├── bbn_evaluator.py       # Interfaz con camb.bbn y cálculo de chi^2 observacional
│   ├── mcmc_sampler.py        # Muestreo Bayesiano de parámetros con emcee (Fase 2)
│   └── emulator_model.py      # Emulador MLP en PyTorch para aceleración MCMC (Fase 3)
├── data/                      # Dataset sintético para entrenamiento de IA
├── tests/                     # Tests unitarios de validación numérica
├── requirements.txt           # Lista de dependencias en Python
└── README.md                  # Documentación del repositorio
Flujo de Trabajo (Workflow)Fase 1 (Evaluación Directa):Computa el perfil analítico $Q(T)$ propuesto por Gabriel León et al.Calcula $\Delta N_{\text{eff}}$ en la escala de desacoplamiento débil.Modifica la predicción de $Y_p$ y $D/H$ mediante camb.bbn y contrasta con los datos observacionales mediante $\chi^2$.Fase 2 (Re-parametrización e Inferencia):Corre cadenas de Markov (MCMC) vía emcee si los parámetros originales de $Q(T)$ generan tensión cosmológica.Define la distribución a posteriori de parámetros compatibles con BBN y cosmología tardía.Fase 3 (Emulación con Deep Learning):Muestrea el espacio de parámetros mediante Latin Hypercube Sampling.Entrena un Perceptrón Multicapa (MLP) en PyTorch para mapear parámetros de $Q \to \{Y_p, D/H\}$ en milisegundos.Instalación y Uso1. Clonar el repositorio y crear el entorno virtualBashgit clone [https://github.com/tu-usuario/ug-diffusion-bbn.git](https://github.com/tu-usuario/ug-diffusion-bbn.git)
cd ug-diffusion-bbn
python -m venv venv
venv\Scripts\activate  # En Windows
2. Instalar dependenciasBashpip install -r requirements.txt
3. Ejecutar la evaluación de la Fase 1Bashpython src/bbn_evaluator.py
Datos Observacionales de Referencia (PDG / Planck)Helio-4 ($Y_p$): $0.245 \pm 0.003$Deuterio ($D/H$): $(2.54 \pm 0.04) \times 10^{-5}$ReferenciasLeón, G., et al. — Unimodular Gravity and Diffusion Terms in Cosmology.Pitrou, C., et al. (2018) — Precision Big-Bang Nucleosynthesis with PRIMAT.Lewis, A., & Challinor, A. — CAMB: Code for Anisotropies in the Microwave Background.