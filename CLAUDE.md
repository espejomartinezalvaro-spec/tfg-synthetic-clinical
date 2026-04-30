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

## Estado actual del trabajo (actualizar al abrir conversación)

### Redacción (LaTeX — `Redaccion/`)
| Cap | Título                              | Estado                                           |
|-----|-------------------------------------|--------------------------------------------------|
| 1   | Introducción                        | Esqueleto creado (pendiente redactar al final)   |
| 2   | Estado del Arte                     | Completado                                       |
| 3   | Dataset y Preprocesamiento          | Completado                                       |
| 4   | Modelos Generativos                 | Pendiente — siguiente a redactar                 |
| 5   | Evaluación de Fidelidad             | Pendiente                                        |
| 6   | Utilidad Analítica                  | Pendiente                                        |
| 7   | Análisis de Privacidad              | Pendiente                                        |
| 8   | Discusión y Conclusiones            | Esqueleto reestructurado (§8.1 Discusión / §8.2 Conclusiones) |

### Notebooks ejecutados (`notebooks/`)
| Notebook                       | Estado     | Output clave                                       |
|--------------------------------|------------|----------------------------------------------------|
| 01_eda_exploracion_inicial     | Ejecutado  | EDA inicial, distribuciones, missingness           |
| 02_preprocessing               | Ejecutado  | Snapshot (22 520 × 127), tensor (22 235 × 48 × 10) |
| 03_data_validation             | Ejecutado  | Validación de datasets procesados                  |
| 04_train_ctgan_tvae            | Creado     | Pendiente ejecución (servidor)                     |
| 05_train_tabddpm               | Creado     | Pendiente ejecución (servidor)                     |
| 06_train_timegan               | Creado     | Pendiente ejecución (servidor) — ver decisión pendiente |
| 07_train_dp_model              | Creado     | Pendiente ejecución (servidor)                     |

---

## Decisión pendiente: alcance de modelos

**Modelos confirmados:** CTGAN · TVAE · TabDDPM · DP-CTGAN (Opacus, epsilon sweep {1,5,10,∞})

**TimeGAN — pendiente de decisión:**
- Es el único modelo temporal; su eliminación suprime toda la rama de series temporales
- Riesgo real: convergencia inestable, entrenamiento en 3 fases, requiere GPU en servidor
- Opción A: eliminar — scope más limpio, menor riesgo, pero el tensor 22 235×48×10 queda sin uso
- Opción B: mantener como módulo secundario — análisis más ligero, sin DP, sin comprometer el núcleo
- Si se elimina, hay ajustes necesarios en §3.5 y en §2 (véase sección "Impacto de eliminar TimeGAN")

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
- **Fase 1 (instalado):** pandas, numpy, matplotlib, seaborn, scikit-learn, tqdm, pyarrow, ipykernel
- **Fase 2+ (servidor):** PyTorch + Opacus, SDV/SDMetrics, XGBoost, UMAP-learn, MLflow, DVC
- **LaTeX:** XeLaTeX, plantilla FIUM, bibstyle `unsrtnat`, paquete natbib

---

## Datos clave del dataset (MIMIC-III)

- Cohorte final: **22 520 estancias UCI** (filtro: primera estancia, LOS ≥ 48h, edad ≥ 18)
- Mortalidad: **13,8 %** · Edad media: **65,1 años** · Periodo: 2001–2012
- Variables vitales (11): heart_rate, sbp, dbp, mbp, spo2, temp_c, resp_rate, gcs_eye/verbal/motor
- Biomarcadores lab (18): creatinina, lactato, glucosa, hemoglobina, plaquetas, bilirrubina, sodio,
  potasio, bicarbonato, wbc, ph_arterial, pao2, paco2, exceso_base, troponina, inr, bun, albúmina
- Representación tabular: snapshot 48h → parquet **(22 520 × 127)**
- Representación temporal: tensor numpy **(22 235 × 48 × 10)** — pendiente de decisión sobre uso

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

## Impacto de eliminar TimeGAN (referencia rápida)

Si se decide eliminar TimeGAN, estos son los ajustes necesarios:

**`3_dataset_preprocesamiento.tex`**
- §3.5 §3.5.2 "Series temporales horarias": eliminar o convertir en trabajo futuro
- §3.5.3 "Coherencia entre representaciones": eliminar (no tiene sentido sin dos representaciones)
- §3.5.1 "Snapshot tabular": queda como sección única, renombrar §3.5 a "Representación tabular"
- §3.5.2 (zero-inflation GCS): el párrafo sobre intubados y bimodalidad pierde su justificación
  ("el modelo TimeGAN debe…") — reencuadrar o eliminar
- §3.6.1: la frase sobre "modelos generativos condicionales" sigue siendo válida
- Tabla de cohorte: el dato "22 235 estancias en tensor" desaparece

**`2_estado_arte.tex`**
- §2.2: el apartado de TimeGAN (TimeGAN, Yoon et al. 2019) se puede mantener como contexto
  histórico o mover a trabajo futuro — no es obligatorio eliminarlo del estado del arte

**`main.tex` / comentarios**
- Actualizar comentario del cap. 4: quitar "TimeGAN" de la lista

---

## Punto de partida para nueva conversación

Al iniciar una sesión nueva, recordar:
1. ¿En qué sección de redacción nos quedamos? → ver tabla "Estado actual"
2. ¿Hay notebooks recién ejecutados con nuevos outputs? → actualizar tabla de notebooks
3. Resolver la **decisión pendiente sobre TimeGAN** antes de redactar el capítulo 4
4. El siguiente capítulo a redactar es el **Capítulo 4 — Modelos Generativos**
   (CTGAN · TVAE · TabDDPM · DP-CTGAN confirmados; TimeGAN pendiente de decisión)
