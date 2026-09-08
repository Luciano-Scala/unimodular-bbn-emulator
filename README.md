# unimodular-bbn-emulator

Evaluación computacional de la compatibilidad entre el modelo de inflación
por difusión unimodular de León (*Class. Quant. Grav.* 39, 075008, 2022)
y las restricciones observacionales de Nucleosíntesis Primordial (BBN),
usando MCMC acelerado por un emulador de red neuronal.

**Contexto:** este repositorio implementa el trabajo computacional del
Plan de Tesis de Licenciatura de Luciano Scala (director: Dr. Gabriel R.
Bengochea, IAFE-CONICET/UBA), cubriendo los objetivos específicos 2-4 del
plan (estudio del modelo de León, análisis de BBN, comparación con ΛCDM).
El objetivo 1 (fundamentos conceptuales de RG y expansión acelerada) se
desarrolla aparte, fuera de este código.

## Resultado principal

Bajo la restricción de $N_{\rm eff}$ más precisa disponible (Goldstein &
Hill 2026, $N_{\rm eff}=2.990\pm0.070$) y exigiendo además la condición
de transición suave que sostiene el cálculo del espectro primordial del
propio paper de León (su Ec. 21):

- El ansatz original de León (`tanh`, $\alpha=0.027$, $N_f=300$) **no
  concilia BBN con el valor observado de la constante cosmológica hoy**
  bajo ninguna combinación de parámetros explorada.
- Una generalización del ansatz ($\mathrm{sinh}^{2n}$, motivada por la
  nota 8 del propio paper) **sí admite una región viable**, angosta:
  $n \in [0.43, 0.58]$, pico en $n\approx0.47$-$0.48$, con
  $\alpha\approx0.11$, $N_f\approx350$.

Ver `report.tex` para el análisis completo, cifras y figuras.

## Estructura del repositorio

unimodular_physics.py # Motor físico: eps1(N), Q(N), rho(N), mapeo N<->T, Gamma(T)
bbn_evaluator.py # Traducción Gamma(T) -> Delta_Neff -> chi2 (vs. Goldstein & Hill 2026)
mcmc_sampler.py # MCMC (emcee): priors, likelihood, log_posterior, corner plots
dataset_generator.py # Latin Hypercube Sampling + evaluación física -> CSV de entrenamiento
emulator_model.py # MLP (PyTorch): entrena y expone BBNEmulator para inferencia rápida
validate_chi2_emulator.py # Valida el emulador EN EL ESPACIO DE CHI2 (no en Yp/D-H por separado)
find_map.py # Extrae el MAP real de una cadena MCMC guardada (backend HDF5)
joint_constraint_region.py # Barrido de grilla (alpha, Nf): región BBN-ok vs. BBN+Lambda_obs-ok
n_sensitivity_scan.py # Barrido en n (paso grueso) -- ver n_sensitivity_fine.py para el resultado bueno
n_sensitivity_fine.py # Barrido en n de alta resolución, con gráfico (resultado final usado en el reporte)
make_comparison_plot.py # Figura estilo-paper: eps1(N) y Q(N) antes/después
compare_chains.py # Compara cuantiles entre cadena de física exacta y cadena con emulador
report.tex # Reporte final (LaTeX)

dataset_tanh.csv, dataset_sinh2n.csv # Datasets generados (10.000 puntos válidos c/u)
emulator_tanh.pt, emulator_sinh2n.pt # Checkpoints de los emuladores entrenados
chain_tanh_emu.h5, chain_sinh2n_emu.h5 # Cadenas MCMC (backend emcee/HDF5)


## Orden de ejecución (reproducir desde cero)

```bash
# 0. Entorno
pip install numpy scipy matplotlib pandas torch emcee corner h5py camb tqdm

# 1. Validar el motor físico contra los valores reportados en el paper de León
python unimodular_physics.py

# 2. Generar datasets de entrenamiento (LHS + física exacta; ~1-2 min c/u)
python dataset_generator.py --model tanh   --n_valid 10000 --batch_size 1000
python dataset_generator.py --model sinh2n --n_valid 10000 --batch_size 1000

# 3. Entrenar los emuladores (predicen Delta_Neff directamente; ~1-2 min c/u en CPU)
python emulator_model.py --model tanh   --csv dataset_tanh.csv
python emulator_model.py --model sinh2n --csv dataset_sinh2n.csv

# 4. Validar el emulador EN EL ESPACIO DE CHI2 (paso obligatorio, no opcional:
#    el error relativo en Yp/D-H o en Delta_Neff por separado puede ser
#    engañoso cerca de valores nulos -- ver nota en emulator_model.py)
python validate_chi2_emulator.py --model tanh   --csv dataset_tanh.csv
python validate_chi2_emulator.py --model sinh2n --csv dataset_sinh2n.csv

# 5. MCMC (usa el emulador; ~4-5 min por modelo, 32 walkers x 5000 pasos)
python mcmc_sampler.py

# 6. Análisis del posterior
python find_map.py --model tanh
python find_map.py --model sinh2n
python joint_constraint_region.py
python n_sensitivity_fine.py
python make_comparison_plot.py

# 7. Compilar el reporte
pdflatex report.tex && pdflatex report.tex   # o subir a Overleaf
```

**Importante:** si se modifica cualquier definición de prior o likelihood
(`SIGMA_LATE_DEX`, `EPS2_MAX`, bounds de `MODEL_SPECS`, la fórmula de
`bbn_chi2`), **borrar los archivos `chain_*_emu.h5` antes de re-correr**
`mcmc_sampler.py`. Por defecto el sampler resume cadenas existentes
(`resume=True`), lo que mezclaría muestras generadas bajo definiciones de
posterior distintas si no se limpia antes.

```bash
Remove-Item chain_tanh_emu.h5, chain_sinh2n_emu.h5 -ErrorAction SilentlyContinue  # PowerShell
# rm chain_tanh_emu.h5 chain_sinh2n_emu.h5                                        # bash
```

## Decisiones metodológicas clave (y por qué)

- **Observable de BBN:** se usa $N_{\rm eff}$ combinado de Goldstein &
  Hill (2026, arXiv:2603.13226) en vez de $Y_p$/D-H por separado, porque
  ya combina de forma óptima $Y_p$ (LBT $Y_p$ Project V, Yeh et al. 2026),
  D/H, CMB y BAO. Evaluado en $T=0.75$ MeV (congelamiento débil), el canal
  dominante de sensibilidad.
- **Prior de suavidad** (`EPS2_MAX=0.3` en `mcmc_sampler.py`): exige
  $|d\ln\epsilon_1/dN|\ll1$ durante inflación (Ec. 21 de León), necesario
  para que el cálculo del espectro primordial casi-invariante de escala
  del paper original siga siendo válido. Sin este chequeo, el ajuste
  óptimo de BBN+$\Lambda_{\rm obs}$ cae en transiciones tipo escalón
  ($\alpha\to$ borde del prior) que violan esta condición.
- **Emulador validado en el espacio de $\chi^2$, no en error relativo de
  las salidas crudas:** con $\Delta N_{\rm eff}$ como target, el error
  relativo diverge artificialmente cerca de la meseta donde
  $\Delta N_{\rm eff}\approx0$, sin que eso afecte al $\chi^2$ real
  (que depende de un error absoluto fijo, $\sigma_{\rm obs}=0.070$). Ver
  `validate_chi2_emulator.py`.
- **Resolución de grilla en barridos de `n`:** la región viable final
  resultó más angosta ($\Delta n\approx0.15$) que el paso de grilla
  inicial ($\Delta n\approx0.25$), lo que producía falsos negativos
  (`n_sensitivity_scan.py`). `n_sensitivity_fine.py` (paso
  $\Delta n\approx0.025$) es la versión correcta/definitiva.

## Limitaciones conocidas

- **Aproximación de un punto de referencia en $T$**: no se integra sobre
  toda la ventana de BBN con una red nuclear completa. El siguiente paso
  natural es una integración con AlterBBN nativo (o PRIMAT), inyectando
  $H_{\rm UG}(T)$ completo en vez de la traducción a $\Delta N_{\rm eff}$
  puntual.
- **Umbral `EPS2_MAX=0.3`**: interpretación razonable de "≪1" (Ec. 21 del
  paper), no derivada cuantitativamente. Un umbral más estricto
  angostaría aún más la región viable de `sinh2n`.
- **Proxy fijo de "hoy"** (`N0_MINUS_NF_PROXY=62.3` en
  `unimodular_physics.py`): calibrado sobre el ansatz `tanh` original; no
  se recalcula el matching completo materia-radiación (Sec. V de León)
  para cada punto muestreado.
- **Extensión post-inflacionaria de `sinh2n`**: la cola de saturación
  para $N>N_f$ (necesaria para evaluar $\Gamma$ en BBN) es una elección
  de modelado propia, no derivada del paper original.

## Referencias clave

- G. León, *"Inflation and the cosmological (not-so) constant in
  unimodular gravity"*, Class. Quant. Grav. 39, 075008 (2022),
  arXiv:2202.04029.
- S. Goldstein, J. C. Hill, *"A 2% determination of $N_{\rm eff}$..."*,
  arXiv:2603.13226 (2026).
- T.-H. Yeh et al., *"The LBT $Y_p$ Project V"*, arXiv:2601.22239 (2026).