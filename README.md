# Generación de Datos Clínicos Sintéticos mediante Modelos Generativos Profundos

**Evaluación de la Privacidad y la Utilidad Analítica sobre el Dataset MIMIC-III**

Trabajo Fin de Grado · Grado en Ciencia e Ingeniería de Datos  
Facultad de Informática · Universidad de Murcia · 2025–2026

**Autor:** Álvaro Espejo Martínez  
**Tutores:** José Manuel Juárez Herrero · Bernardo Cánovas Segura
**Calificación**: 9,5 / 10 — Sobresaliente
---

## Descripción

Este proyecto propone y evalúa un marco experimental de generación de datos clínicos sintéticos mediante modelos generativos profundos entrenados sobre la cohorte UCI de MIMIC-III. Se implementan y comparan cuatro arquitecturas —CTGAN, TVAE, TabDDPM y DP-CTGAN— evaluadas en tres dimensiones complementarias: fidelidad estadística, utilidad analítica para tareas de predicción clínica y riesgo de privacidad, con especial atención al tradeoff privacidad–utilidad bajo Differential Privacy.

---

## Estructura del repositorio

```
TFG/
├── data/
│   ├── raw/          ← tablas MIMIC-III originales (.csv.gz)   NO versionado (DUA)
│   ├── interim/      ← pasos intermedios del preprocesamiento
│   └── processed/    ← datasets finales (tabular_48h.parquet, tensor secuencial)
├── notebooks/        ← pipeline completo en Jupyter (ver tabla más abajo)
├── src/
│   ├── data/         ← scripts de extracción y preprocesamiento
│   ├── models/       ← implementaciones de los modelos generativos
│   └── evaluation/   ← métricas de fidelidad y privacidad
├── reports/          ← figuras y tablas exportadas para la memoria
├── Redaccion/        ← memoria en LaTeX (XeLaTeX, plantilla FIUM/UMU)
│   ├── main.tex
│   ├── bibliografia/
│   └── *.tex         ← un fichero por capítulo
└── requirements.txt
```

---

## Requisitos

- Python 3.12
- CUDA compatible con PyTorch 2 (entrenamiento en GPU RTX A4500)

```bash
pip install -r requirements.txt
```

---

## Acceso a los datos (MIMIC-III)

Los datos de MIMIC-III son de **acceso controlado** y no están incluidos en este repositorio. Para obtenerlos:

1. Completar el curso de ética CITI Program (módulo *Data or Specimens Only Research*)
2. Firmar el *Data Use Agreement* (DUA) con PhysioNet: https://physionet.org/content/mimiciii/
3. Descargar las tablas necesarias y colocarlas en `data/raw/`

Las tablas utilizadas son: `ADMISSIONS`, `ICUSTAYS`, `PATIENTS`, `CHARTEVENTS`, `LABEVENTS`, `D_ITEMS`, `D_LABITEMS`.

>  **Nunca subir `data/raw/` a ningún repositorio público.** Es una obligación del DUA.

---

## Pipeline de notebooks

| Notebook | Descripción | Output principal |
|---|---|---|
| `01_eda_exploracion_inicial` | EDA inicial, distribuciones, missingness | Figuras exploratorias |
| `02_preprocessing` | Extracción, limpieza, imputación, agregación | `tabular_48h.parquet` (22 520 × 127) |
| `03_data_validation` | Validación de rangos fisiológicos y consistencia | Informe de validación |
| `04_train_ctgan_tvae` | Entrenamiento de CTGAN y TVAE (SDV) | `ctgan_samples.parquet`, `tvae_samples.parquet` |
| `05_train_tabddpm` | Entrenamiento de TabDDPM (PyTorch) | `tabddpm_samples.parquet` |
| `05b_tabddpm_ablation` | Experimento de ablación de normalización | `tabddpm_v2_samples.parquet` (descartado) |
| `07_train_dp_model` | Entrenamiento de DP-CTGAN con Opacus (ε ∈ {1,5,10,∞}) | `dp_ctgan_ε*.parquet` |
| `08_evaluation_fidelity` | Evaluación de fidelidad: JSD, KS, MMD, Wasserstein, UMAP, t-SNE | `reports/fidelidad_summary.csv` + figuras |
| `09_evaluation_utility` | Evaluación de utilidad: TRTR/TSTR mortalidad, augmentation, sepsis | `reports/utilidad_summary.csv` + figuras |
| `10_evaluation_privacy` | Evaluación de privacidad: DCR, NNDR, MIA, Pareto | `reports/privacidad_summary.csv` + figuras |

> El notebook `06_train_timegan` está excluido del alcance experimental (representación temporal fuera de scope).

---

## Resultados principales

| Modelo | JSD med. | AUROC TSTR | MIA advantage |
|---|---|---|---|
| TVAE | 0,0085 | 0,835 | +0,297 |
| CTGAN | 0,0080 | 0,778 | +0,271 |
| TabDDPM | 0,6442 | 0,780 | +0,199 |
| DP-CTGAN ε=∞ | 0,0229 | 0,744 | +0,300 |
| DP-CTGAN ε=10 | 0,0424 | 0,692 | +0,284 |
| DP-CTGAN ε=5 | 0,0506 | 0,713 | +0,274 |
| DP-CTGAN ε=1 | 0,1057 | 0,663 | +0,221 |

Referencia holdout real: MIA advantage +0,390. Ningún modelo la supera.  
Frontera Pareto DP-CTGAN: reducir ε de ∞ a 1 cuesta 0,081 AUROC y reduce MIA advantage en 0,079.

---

## Memoria

La memoria está redactada en LaTeX con la plantilla oficial FIUM/UMU y compilada en Overleaf. Los fuentes están en `Redaccion/`. Las figuras se generan con los notebooks y se exportan a `reports/` para su inclusión manual en Overleaf.

---

## Licencia

El código de este repositorio se distribuye bajo licencia MIT.  
Los datos de MIMIC-III están sujetos al *Data Use Agreement* de PhysioNet y no pueden redistribuirse.
