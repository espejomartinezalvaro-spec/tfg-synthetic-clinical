# CLAUDE.md — TFG: Generación de Datos Clínicos Sintéticos

## Contexto del proyecto

**Título:** Generación de Datos Clínicos Sintéticos mediante Modelos Generativos Profundos:
Evaluación de la Privacidad y la Utilidad Analítica sobre el Dataset MIMIC-III

**Autor:** Álvaro Espejo Martínez · Universidad de Murcia (FIUM) · Grado en Ciencia e Ingeniería de Datos
**Tutores:** José Manuel Juárez Herrero y Bernardo Cánovas Segura

**Objetivo general:** Desarrollar y evaluar un marco de generación de datos biomédicos sintéticos
mediante GANs, VAEs y Modelos de Difusión sobre MIMIC-III, garantizando fidelidad estadística,
utilidad analítica y cuantificación del tradeoff privacidad–fidelidad mediante Differential Privacy.

---

## Estructura de la memoria (6 capítulos — estructura vigente)

```
main.tex
├── Cap 1 — Introducción                    (esqueleto, redactar al final)
├── Cap 2 — Estado del Arte                 (COMPLETO)
├── Cap 3 — Objetivos y Metodología         (COMPLETO)
├── Cap 4 — Desarrollo
│   ├── §4.1 Dataset y Preprocesamiento     (COMPLETO)
│   └── §4.2 Modelos Generativos            (COMPLETO — CTGAN, TVAE, TabDDPM, DP-CTGAN)
├── Cap 5 — Experimentación y Resultados
│   ├── §5.1 Fidelidad Estadística
│   │   ├── §5.1.1–5.1.4 Metodología        (COMPLETO)
│   │   └── §5.1.5 Discusión comparativa    (PENDIENTE — datos disponibles en nb08)
│   ├── §5.2 Utilidad Analítica             (esqueleto — pendiente nb09)
│   └── §5.3 Análisis de Privacidad         (esqueleto — pendiente nb10)
└── Cap 6 — Discusión y Conclusiones        (esqueleto, redactar al final)
```

Ficheros LaTeX en `Redaccion/`: `1_introduccion.tex`, `2_estado_arte.tex`,
`3_objetivos_metodologia.tex`, `4_desarrollo.tex`, `5_experimentacion_resultados.tex`,
`6_conclusiones.tex`.

---

## Estado de los notebooks

| Notebook                    | Estado     | Output clave                                              |
|-----------------------------|------------|-----------------------------------------------------------|
| 01_eda_exploracion_inicial  | Ejecutado  | EDA inicial, distribuciones, missingness                  |
| 02_preprocessing            | Ejecutado  | Snapshot (22 520 × 127), tensor (22 235 × 48 × 10)        |
| 03_data_validation          | Ejecutado  | Validación de datasets procesados                         |
| 04_train_ctgan_tvae         | Ejecutado  | ctgan_samples.parquet, tvae_samples.parquet               |
| 05_train_tabddpm            | Ejecutado  | tabddpm_samples.parquet (QuantileTransformer, 1000 épocas)|
| 06_train_timegan            | Descartado | TimeGAN excluido del scope (ver nota abajo)               |
| 07_train_dp_model           | Ejecutado  | dp_ctgan_ε{1,5,10,inf}_samples.parquet                    |
| 08_evaluation_fidelity      | Ejecutado  | reports/fidelidad_summary.csv + figuras                   |
| 09_evaluation_utility       | Pendiente  | TRTR/TSTR mortalidad + sepsis — necesario para §5.2       |
| 10_evaluation_privacy       | Pendiente  | MIA, DCR, NNDR — necesario para §5.3                      |

---

## Resultados de fidelidad (nb08 — definitivos)

| Modelo        | JSD med | KS med | MMD²   | W1       | Δρ    |
|---------------|---------|--------|--------|----------|-------|
| TVAE          | 0.0085  | 0.082  | 0.0053 | 19.9     | 0.035 |
| CTGAN         | 0.0080  | 0.117  | 0.0189 | 20.6     | 0.059 |
| DP-CTGAN ε=∞  | 0.0251  | 0.159  | 0.0256 | 43.7     | 0.066 |
| DP-CTGAN ε=10 | 0.0400  | 0.159  | 0.0121 | 32.2     | 0.144 |
| DP-CTGAN ε=5  | 0.0565  | 0.219  | 0.0275 | 28.8     | 0.173 |
| TabDDPM       | 0.6442  | 0.530  | 0.5541 | 1359.3   | 0.056 |
| DP-CTGAN ε=1  | 0.1157  | 0.299  | 0.1191 | 104.6    | 0.421 |

AUC discriminador = 1.0 para todos los modelos (XGBoost demasiado potente — resultado esperado).
TabDDPM: JSD/W1/MMD² altos pero Δρ bueno — preserva correlaciones pero no distribuciones marginales.
DP-CTGAN: degradación monotónica en JSD y Δρ al reducir ε — resultado científico central del TFG.

---

## Decisiones de alcance ya tomadas

- **TimeGAN: excluido del scope.** La representación temporal (tensor 22 235×48×10) queda fuera del
  alcance experimental. Mencionado en §4.2 como trabajo futuro. No hay que retomar esta decisión.
- **Modelos evaluados:** CTGAN · TVAE · TabDDPM · DP-CTGAN (ε ∈ {1, 5, 10, ∞})
- **TabDDPM normalización:** QuantileTransformer(output='normal', n_quantiles=1000) — no StandardScaler.
  Hiperparámetros reales: cosine schedule, hidden_dims=(512,512,512,512), 1000 épocas, lr=3e-4.
  ⚠️ La tabla §4.2 (tab:hp_tabddpm) tiene inconsistencias con la implementación real — pendiente corregir.

---

## Estructura de carpetas

```
TFG/
├── data/
│   ├── raw/          ← tablas MIMIC-III .csv.gz  ⚠️ EN .gitignore, NUNCA versionar
│   ├── interim/      ← pasos intermedios
│   └── processed/    ← datasets finales (parquet, numpy)
├── notebooks/        ← jupyter notebooks de exploración y entrenamiento
├── src/
│   ├── data/         ← scripts extracción y preprocesamiento
│   ├── models/       ← código modelos generativos
│   └── evaluation/   ← métricas fidelidad y privacidad
├── reports/          ← figuras y tablas para la memoria
├── Redaccion/        ← LaTeX: main.tex + capítulos + bibliografía
│   ├── main.tex
│   ├── bibliografia/
│   │   └── bibliografia.bib
│   └── *.tex         ← un fichero por capítulo
└── CLAUDE.md         ← este fichero
```

---

## Stack tecnológico

- **Python 3.12** en Windows nativo (sin WSL)
- **Fase 1:** pandas, numpy, matplotlib, seaborn, scikit-learn, tqdm, pyarrow, ipykernel
- **Fase 2+:** PyTorch + Opacus, SDV/SDMetrics, XGBoost, UMAP-learn
- **LaTeX:** XeLaTeX, plantilla FIUM, bibstyle `unsrtnat`, paquete natbib

---

## Datos clave del dataset (MIMIC-III)

- Cohorte final: **22 520 estancias UCI** (filtro: primera estancia, LOS ≥ 48h, edad ≥ 18)
- Mortalidad: **13,8 %** · Edad media: **65,1 años** · Periodo: 2001–2012
- Variables vitales (10): heart_rate, sbp, dbp, mbp, spo2, temp_c, resp_rate, gcs_eye/verbal/motor
- Biomarcadores lab (18): creatinina, lactato, glucosa, hemoglobina, plaquetas, bilirrubina, sodio,
  potasio, bicarbonato, wbc, ph_arterial, pao2, paco2, exceso_base, troponina, inr, bun, albúmina
- Representación tabular: snapshot 48h → parquet **(22 520 × 127)**

---

## Convenciones de código

- Rutas siempre con `pathlib.Path` relativas a la raíz del proyecto
- `random_state=42` en todos los modelos
- Prefijos de commit: `feat:` `fix:` `data:` `docs:` `exp:`
- **No incluir** la línea `Co-Authored-By: Claude` en los commits
- **Nunca** subir `data/raw/` ni `token.txt` a Git (obligación DUA de MIMIC)

---

## Reglas para la redacción LaTeX

- Todo enunciado factual debe llevar `\cite{}` (única excepción: ideas propias)
- No modificar secciones ya redactadas salvo que el usuario lo pida explícitamente
- Seguir el estilo del cap. 2: `\paragraph{}`, `\textbf{}`, ecuaciones numeradas, citas natbib
- Redactar **sección a sección** ("de poco en poco"), no el capítulo entero de una pasada
- Archivo de bibliografía: `Redaccion/bibliografia/bibliografia.bib`

---

## Punto de partida para nueva conversación

1. La memoria tiene **6 capítulos** (no 8). Ver tabla de estructura arriba.
2. Cap 2, 3 y 4 están **completamente redactados**.
3. Lo siguiente a redactar es **§5.1.5** (discusión comparativa de fidelidad) — datos en nb08.
4. Antes de §5.2 y §5.3 hay que crear y ejecutar nb09 y nb10 en el servidor.
5. La tabla `tab:hp_tabddpm` en §4.2 tiene inconsistencias con la implementación — pendiente corregir.
6. TimeGAN: **decisión ya tomada** — excluido, mencionado como trabajo futuro en §4.2.
