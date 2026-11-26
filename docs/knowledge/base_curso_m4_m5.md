# Base de conocimiento — Módulos 4 y 5

Fuente: PDFs en `Recursos/`. Cada afirmación operativa cita `PDF p.N` del archivo de la unidad. Las páginas sin texto OCR se marcan como diapositiva de imagen.

Inventario: 9 PDF, 435 páginas. Falta la Unidad 2 del Módulo 5 en la carpeta.

## Cómo se usa esta base en la Fase III

El modelo de la entrega es un bi-encoder Transformer (codificador MiniLM) entrenado como red siamesa con pérdida contrastiva / InfoNCE. Eso ancla M5U4 (Transformers) y M5U5 (siamesas).
TF-IDF, BM25 y la fusión lineal son baselines del Módulo 4 (embeddings clásicos y similitud léxica). Se miden en la pregunta sin extracto pegado.
Train / desarrollo / gold sigue M4U4 p.23: hiperparámetros en validación, desempeño en prueba.
CNN (M5U3) y generativos de imagen (parte de M5U5) quedan fuera del ranking CTI.

## Módulo 4 · Unidad 1 — Introducción a NLP

**Archivo:** `MLDS4_U1_IntroNLP_2026-1.pdf` (39 páginas; 18 con texto).

### Objetivo

Definir el lenguaje natural, sus ambigüedades y por qué NLP no se reduce a buscar cadenas exactas.

### Conceptos (texto extraíble)


- **p.18.** •Total:estaimplicaquelaspalabrastienenelmismosignificadoenuncontexto. •Parcial:cuandolaspalabrastienensignificadossimilaresendiferentesgradoscambiandolevementeelsignificadodelafrase.AmbigüedadRetos de NLP •"Ese veneno es letal“ •"Ese veneno es mortífero"Sinonimia total •"Hablé co…
- **p.19.** •Ellenguajenaturalsuponequeexisteunabasedeconocimientocomúnentrequienessecomunican;comparteninformacióngeneralacercadeunagrancantidaddetemas,locualpermiteunacomunicaciónfluidayunentendimientomutuo. •Sinembargo,estorepresentaunagrandificultadparaunsistemaquenocuentaconunconocimien…
- **p.20.** •Enlalenguahabladanosesuelenhacerpausasentrepalabraypalabra. •Ellugarenelquesedebesepararlaspalabrasamenudodependedelaposibilidadquelafrasemantengaunsentidológicotantoanivelgramaticalcomocontextual. •Enlalenguaescrita,lenguascomoelchinomandaríntampocotienenseparacionesentrelaspal…
- **p.21.** ♥1Problemas de segmentación2Expresiones •Estácomounacabra •Notienenipiesnicabeza •Breaktheice •Speakofthedevil •Undertheweather3NeologismosExpresionesopalabrasrecientesqueprovienendeunalenguaextranjeraeingresanaotroidioma.Enespañol: •Pósit(postit) •Friendzone •Retweet •Like4Nombr…
- **p.22.** •Elprocesamientodelenguajenaturalseusaampliamenteenlaactualidad. •Podemosencontraraplicacionesdesdeelautocompletadodeloscelulares,hastaasistentesvirtualescomoAlexa,Siri,GoogleAssistant,ChatGPTydemás.

Páginas 1–16, 23–38: en su mayoría diapositivas de imagen o créditos (sin OCR usable).

### Implicación para el proyecto

La pregunta del analista está en español y el índice CVE/MITRE está en inglés. Sinonimia y conocimiento del mundo (p.18–19) explican por qué un léxico puro falla si no comparte tokens. El asistente no inventa hechos: recupera y cita.

### Evidencia en el repo

- Preguntas en español: `docs/data/evaluation/gold_set_es.json`.
- Corpus endurecido en inglés: `data/processed/*_hardened.parquet`.
- Abstención y citas: `src/threatintel/rag/generate.py`.

## Módulo 4 · Unidad 2 — Preprocesamiento

**Archivo:** `MLDS4_U2_Preprocesamiento - 2026-1.pdf` (32 páginas; 5 con texto).

### Objetivo

Preparar el corpus antes del modelado: entender el negocio, el corpus y limpiarlo.

### Conceptos


- **p.5.** •Entendimientodelnegocio •Entendimientodelcorpus •Preparacióndelcorpus •Modelamiento •Evaluación •Despliegue

El resto del PDF es casi todo imagen. El ciclo mostrado en p.5 es el de CRISP/TDSP aplicado a NLP: negocio → corpus → preparación → modelado → evaluación → despliegue.

### Implicación

En este proyecto la preparación es el hardening CTI: detection STIX completo, táctica enriquecida, separación `*_clean` / `*_hardened`, y chunks con preámbulo.

### Evidencia

- `scripts/preprocessing_eda/harden_fase2_cti.py`
- `src/threatintel/rag/lineage_gate.py`
- `src/threatintel/rag/corpus.py`

## Módulo 4 · Unidad 3 — Embeddings

**Archivo:** `MLDS4_U3_Embeddings_2026-1.pdf` (33 páginas; 5 con texto).

### Objetivo

Pasar de conteos a representaciones vectoriales que capturen significado y permitan similitud.

### Conceptos


- **p.8.** ✅ ✖️Escalabilidad computacional ✅ ✖️Contexto ✖️ ✅Sinonimia ✖️ ✅

La diapositiva p.8 compara representaciones: escalabilidad computacional frente a contexto y sinonimia. El resto del PDF no aportó OCR; el curso cubre BoW/TF-IDF, word2vec/Doc2Vec y el paso a embeddings contextuales (lo que en Fase III se usa vía sentence-transformers).

### Implicación

TF-IDF + coseno y BM25 son el baseline léxico. El bi-encoder aporta el embedding contextual multilingüe. Doc2Vec de Fase II queda como control didáctico, no como ranking de producto.

### Evidencia

- Baseline MiniLM: `src/threatintel/retrieval_baseline.py`
- TF-IDF / BM25 / fusión lineal: `src/threatintel/rag/classical.py`
- Informe: `docs/modeling/modelos_clasicos.md`

## Módulo 4 · Unidad 4 — Análisis supervisado

**Archivo:** `MLDS4_U4_Análisis_supervisado_2026-1.pdf` (35 páginas).

### Objetivo

Modelar tareas con etiquetas humanas: el modelo toma un embedding y automatiza una decisión (p.5).

### Conceptos


- **p.5.** Etapa actual del ciclo de vida en NLP El modelamiento es una etapa fundamental en proyectos de NLP. Un modelo toma una representación de los textos (embedding) y permite automatizar distintas tareas relacionadas a los lenguajes naturales. Adaptado de CRISP-DM Cross Industry Stand…
- **p.7.** Es posible utilizar modelos matemáticos y algorítmicos para automatizar tareas del lenguaje. El modelamiento normalmente se suele abordar desde dos tipos de enfoques: • modelos supervisados • modelos no supervisados Modelamiento
- **p.9.** Los modelos supervisados en NLP buscan replicar una acción o las decisiones tomadas por humanos. Normalmente requiere un corpus con documentos etiquetados (el resultado de lo que se busca obtener). Análisis Supervisado
- **p.10.** El aprendizaje supervisado en NLP se suele abordar desde distintas dimensiones: Nivel de secuencia Nivel de documento Nivel de corpus Existe una etiqueta asociada a una secuencia de caracteres, por ejemplo, para clasificar tokens, oraciones o segmentar fragmentos de textos. Exist…
- **p.18.** Question – Answering Un modelo puede generar la respuesta ante un documento que contenga una pregunta. Questing answering con Bing Chat Nivel de documento - Aplicaciones Análisis Supervisado
- **p.23.** Validación del modelo Aprendizaje. Construcción del modelo 1 3 4 Validación del Modelo Análisis Supervisado Datos transformados usando correspondiente representación ( embeddings, grafos,etc) de acuerdo a la aplicación Datos de entrenamiento, con los cuales el modelo aprende. Dat…
- **p.25.** sklearn Herramientas para el Análisis Supervisado Importamos el modelo 1 Obtenemos predicciones del modelo (test) 6 Entrenamos el modelo con un corpus (train) 5 Definimos un modelo 4 Importamos la función para realizar la validación cruzada 2 Particionamos el conjunto de datos 3
- **p.27.** Transformers Herramientas para el Análisis Supervisado Cargamos un modelo 2 Obtenemos las predicciones 3 Importamos la librería 1 HugginFace: empresa para PLN que desarrolla herramientas. Librería Transformers HugginFace Hub: inspirado en github para compartir código, conjunto de…
- **p.30.** Métricas de evaluación CONDICIÓN (Real) TOTAL POPULATION CONDITION POS CONDITION NEG PREVALENCE TEST OUTCOME (Predicción) TEST POS True Pos TP Type I Error False Pos FP Precision Pos Predictive Value False Discovery Rate TEST NEG Type II Error False Neg FN True Neg TN False Omiss…

### Implicación

Question–Answering sobre un documento (p.18) es el patrón del asistente: pregunta + contexto recuperado. La validación hold-out y la separación train/val/test (p.23) obligan a elegir hiperparámetros en desarrollo y reportar el gold una sola vez. Las métricas de ranking (MRR, Recall@k) sustituyen accuracy de clasificación porque la etiqueta es el `doc_id` relevante.

### Evidencia

- Splits: `src/threatintel/rag/pairs.py`
- Grilla en dev: `artifacts/fase3/grid_rows.json`
- Gold hold-out: `docs/data/evaluation/fase3_summary.json`

## Módulo 4 · Unidad 5 — Análisis no supervisado

**Archivo:** `MLDS4_U5_Análisis_no_supervisado - 2026-1.pdf` (48 páginas).

### Objetivo

Encontrar estructura en el corpus sin etiquetas: clustering, tópicos y, con deep learning, similitud semántica y zero-shot.

### Conceptos


- **p.7.** El análisis no supervisado consiste en encontrar patrones de forma automática sin la necesidad de replicar una tarea dada por un humano. A diferencia del análisis supervisado, este tipo de modelos aprenden únicamente a partir del corpus (sin etiquetas). Análisis no supervisado
- **p.8.** Las tareas no supervisadas en NLP involucran agrupamiento y modelado de tópicos. El deep learning ha permitido explorar otro tipo de aplicaciones más modernas. Modelos Análisis no supervisado
- **p.10.** El agrupamiento de textos es una tarea típica en NLP que utiliza técnicas de clustering sobre embeddings. Es una tarea que busca encontrar grupos de documentos que cumplan las siguientes dos reglas: • Los documentos en un mismo grupo deben ser similares. • Los documentos en grupo…
- **p.23.** Modelos de Tópicos Los modelos de tópicos más populares son LSA y LDA. Cada documento se representa como una mezcla de temas. Cada tema se representa como una mezcla de palabras clave. Los modelos de tópicos permiten encontrar e interpretar temáticas dentro de un corpus.
- **p.24.** Latent Semantic Analysis (LSA) Modelos de Tópicos Latent Semantic Analysis (LSA) es un modelo basado en descomposición en valores y vectores propios. Permite obtener los siguientes elementos: Importancia de cada tópico en el corpus. Importancia de cada término dentro del tópico. …
- **p.29.** Latent Dirichlet Allocation (LDA) Modelos de Tópicos Latent Dirichlet Allocation (LDA) es un modelo probabilístico que extiende a LSA. En LDA se aprenden matrices de probabilidad, lo que es más interpretable Distribución documento – tópico Muestra la distribución de probabilidad …
- **p.33.** Comparativa Modelos de Tópicos Característica LSA LDA Baja complejidad computacional Interpretabilidad Usa BoW Usa TF-IDF
- **p.35.** Otras aplicaciones Los recientes avances en deep learning han permitido explorar otro tipo de aplicaciones no supervisadas en NLP. Encontramos aplicaciones no supervisadas como modelos de similitud semántica, zero – shot clasification y modelos generativos. Aplicaciones Similitud…
- **p.36.** Similitud Semántica Otras aplicaciones Los modelos de similitud semántica capturan asociaciones entre conceptos sin la necesidad de que el modelo fuera entrenado para esto. Las representaciones de redes neuronales enfocadas en NLP permiten capturar información semántica de forma …
- **p.37.** Zero-Shot Classification Otras aplicaciones Zero-shot classification permite una clasificación con categorías abiertas. Los modelos no se entrenan para un problema en específico, en lugar de esto, estos modelos aprenden a relacionar cualquier categoría con un texto y determinar d…
- **p.38.** Modelos Generativos Otras aplicaciones Los modelos generativos permiten generar distintos textos en lenguaje natural basándose en generadores de números aleatorios. Se trata de modelos que aprenden la estructura de un lenguaje y son capaces de reproducir las reglas del lenguaje.

### Implicación

El retrieval denso es una tarea de **similitud semántica** (p.36): acercar pregunta y documento sin entrenar un clasificador de severidad. LSA/LDA y KMeans no ordenan el asistente; pueden usarse como exploración de corpus, no como gate de despliegue. Zero-shot (p.37) no sustituye el índice CTI con citas.

### Evidencia

- Coseno sobre embeddings: `src/threatintel/rag/retrieve.py`
- Métricas de ranking: `src/threatintel/eval/metrics.py`

## Módulo 5 · Unidad 1 — Introducción al Deep Learning

**Archivo:** `MLDS5_U1_Introduccion_al_Deep Learning.pdf` (77 páginas; 40 con texto).

### Objetivo

Historia y elementos del deep learning: backpropagation, GPU, ReLU, dropout, transferencia a visión y lenguaje.

### Conceptos (texto extraíble)


- **p.4.** •¿Qué es el aprendizaje profundo? : Generalidades e Historia. •Elementos fundamentales del Deep Learning. •Herramientas para el Deep Learning. •Aplicaciones del Deep Learning. •Tensorflow.FacultaddeProgramade FormaciónMachine Learning and Data ScienceMLDS4
- **p.5.** •¿Qué es el aprendizaje profundo? : Generalidades e Historia. •Elementos fundamentales del Deep Learning. •Herramientas para el Deep Learning. •Aplicaciones del Deep Learning. •Tensorflow.Facultadde MLDS5
- **p.27.** ●Backpropagation ●Backpropagationthroughtime ●Online learning (stochastic gradient descent) ●Softmax(hierarchical)AlgoritmosElementos fundamentales del Deep LearningFacultadde MLDS27
- **p.29.** •De la computación gráfica al procesamiento numérico general (GPGPU). •Arquitectura de instrucción única y datos múltiples. •Cálculos de tipo de alto rendimiento con paralelismo de datos. •Hardware básico. •Dos proveedores principales: NVidia, AMD. •Las unidades de procesamiento …
- **p.35.** ●Layer-wise training ●RELU units ●Dropout ●Adaptive learning rates ●Initialization ●PreprocessingTrucosElementos fundamentales del Deep LearningFacultadde MLDS35
- **p.41.** ●Imagen: Anotación, detección, segmentación, subtitulado ●Vídeo: Seguimiento de objetos, reconocimiento de acciones, segmentaciónVisión por ordenador: ●Imagen: Anotación, detección, segmentación, subtitulado ●Vídeo: Seguimiento de objetos, reconocimiento de acciones, segmentación…
- **p.44.** •Un LargeLanguageModel(LLM) es un tipo de modelo de Machine Learningque abre nuevas posibilidades de comprensión y generación de textos en sistemas de software. Existen varios casos de uso para este modelo, algunos de ellos son : ●Generate: Generación de textos originales y coher…
- **p.46.** •¿Qué es el aprendizaje profundo? : Generalidades e Historia. •Elementos fundamentales del Deep Learning. •Herramientas para el Deep Learning. •Aplicaciones del Deep Learning. •Tensorflow.Facultadde MLDS46
- **p.48.** ●Computaciónnuméricabasadaengrafosdeflujodedatos ●DesarrolladoenC++ ●InterfacesenPythonyC++ ●Diferenciaciónautomática ●FácilvisualizaciónmedianteTensorBoard ●Capasdeabstracción ●Tf.contrib.learn,tf.contrib.slim ●TFLearn,Keras ●Soportedearquitecturasheterogéneas:Multi-CPU,GPU,Mult…

Varias páginas centrales son figura. El hilo útil: el aprendizaje profundo aprende representaciones; el fine-tuning adapta un modelo preentrenado; el texto aparece como dominio de generación, resumen y reescritura (p.44).

### Implicación

No se entrena un Transformer desde cero. Se afina MiniLM con GPU/CPU y se congela lo que no mejora (re-ranker). El despliegue local evita API pública.

### Evidencia

- `src/threatintel/rag/train.py` (SentenceTransformerTrainer, fp16 si CUDA)
- Checkpoint: `artifacts/fase3/biencoder_best`

## Módulo 5 · Unidad 3 — Redes convolucionales

**Archivo:** `v2 MLDS5_U3_Redes_neuronales_convolucionales-v2025-1.pdf` (67 páginas).

### Objetivo

Convolución, pooling, data augmentation, transfer learning y fine-tuning en visión.

### Conceptos


- **p.4.** Agenda • Introducción a Redes Neuronales Convolucionales. • Capas Convolucionales y de Pooling. • Data Augmentation. • Transfer Learning • Fine Tuning. • Transfer Learning y Fine Tuning en Keras
- **p.5.** Agenda • Introducción a Redes Neuronales Convolucionales. • Capas Convolucionales y de Pooling. • Data Augmentation. • Transfer Learning • Fine Tuning. • Transfer Learning y Fine Tuning en Keras
- **p.21.** Pooling Capas Convolucionales y de Pooling Toma el máximo valor en la ventana de muestreo. Max Pooling Toma el promedio de los valores en la ventana de muestreo Average Pooling Toma la suma de los valores en la ventana de muestreo. Sum Pooling La capa de Pooling se aplica después…
- **p.25.** Data Augmentation 3
- **p.33.** Transfer Learning 4
- **p.37.** ¿Cómo funciona? Transfer Learning Funciona transfiriendo tantos conocimientos como sea posible de un modelo existente a un nuevo modelo diseñado para una tarea similar. Se escoge un modelo pre- entrenado en una tarea similar. Se conserva la parte del modelo encargado de aprender …
- **p.43.** Fine Tuning 5
- **p.45.** ¿Cómo funciona? Fine Tuning Se escoge un modelo pre-entrenado. • Se congela el bloque convolucional. • La parte final (que por lo general es un MLP) se entrena durante pocas epochs. Warming Up (calentamiento): Luego, se liberan los pesos de todo el modelo, y se entrena, con un le…

### Implicación

**Fuera de alcance del ranking CTI.** No hay imágenes de entrada en el asistente. La idea de transfer learning / fine-tuning (p.33–45) sí se reutiliza en texto: partimos de un encoder preentrenado y afinamos pares pregunta–documento.

### Evidencia

- Declarado fuera en esta base y en la matriz de rúbrica.
- El fine-tuning textual está en `train_biencoder`.

## Módulo 5 · Unidad 4 — Transformers

**Archivo:** `V2 MLDS5_U4_Transformers.pdf` (55 páginas).

### Objetivo

De modelos de lenguaje probabilísticos y RNN/LSTM a la arquitectura Transformer con autoatención.

### Conceptos


- **p.7.** Modelos de lenguaje Modelos Probabilísticos Traducción automática La oficina está a unos quince miniutos de mi casa. Corrección ortográfica Reconocimiento del habla Los modelos probabilísticos asignan una probabilidad a una secuencia de palabras P(Fuertes vientos esta noche) > P(…
- **p.13.** Infactibilidad de los n-gramas Modelos de lenguaje Complejidad computacional para calcular la probabilidad de cada posible n-grama. Problema
- **p.16.** Redes Neuronales Recurrentes ●Redes neuronales con memoria ●NN feed-forward : La salida depende exclusivamente de la entrada actual. ●NN Recurrente : La salida depende de los estados actuales y anteriores. NNs Recurrentes Esto se consigue mediante conexiones laterales/posteriores…
- **p.21.** Dependencias a largo plazo Redes Neuronales Recurrentes
- **p.22.** Memoria de corto y largo plazo (LSTM) Redes Neuronales Recurrentes Las redes LSTM resuelven el problema de la dependencia a largo plazo Utilizan compuertas que permiten mantener la memoria a través de secuencias largas y actualizarse sólo cuando es necesario. Hochreiter, Sepp, an…
- **p.25.** Módulos de atención Redes Neuronales Recurrentes En tareas de generación de texto, las redes recurrentes hicieron uso de módulos de atención para refinar el resultado.
- **p.29.** Problemas de los modelos Recurrentes Transformers Las RNN se desenrollan "de izquierda a derecha". Esto codifica la localidad lineal: Algo bueno: Las palabras cercanas suelen influir en el significado de las demás.
- **p.31.** Mejor pongamos atención Transformers Todas las palabras atienden a todas las palabras en la capa anterior; aquí omitimos la mayoría de flechas La atención trata la representación de cada palabra como una consulta para acceder e incorporar información de un conjunto de valores. El…
- **p.33.** Transformers - Estructura Transformers Los Transformers tienen dos componentes principales Codificador Decodificador
- **p.34.** Codificador (Encoder) Transformers Todos los codificadores tienen la misma estructura (aunque no comparten pesos). Cada uno se divide en dos subcapas Las salidas de la autoatención se alimentan a una red neuronal Feed-Forward. Se aplica exactamente la misma a cada posición de for…
- **p.36.** Autoatención Transformers “The animal didn't cross the street because it was too tired” Digamos que queremos traducir: ●¿A qué se refiere "it? la autoatención permite asociar it" con "animal". ●La autoatención permite buscar pistas en otras posiciones de la secuencia de entrada q…
- **p.37.** Autoatención – Múltiples cabezas Transformers Amplía la capacidad del modelo para centrarse en distintas posiciones. Dota a la capa de atención de múltiples subespacios de representación
- **p.38.** Codificación Posicional Transformers 01 El transformador añade un vector a cada entrada. Estos vectores siguen un patrón específico que el modelo aprende, lo que le ayuda a determinar la posición de cada palabra, o la distancia entre las distintas palabras de la secuencia. Alamma…
- **p.39.** Decodificador (Decoder) Transformers • El codificador comienza procesando la secuencia de entrada. • El codificador transforma la entrada en un conjunto de vectores de atención. • Cada decodificador se centra en los lugares adecuados de la secuencia de entrada. Alammar, J. (s. f.…

### Implicación

MiniLM es un **codificador** Transformer (p.33–34): autoatención multi-cabeza y codificación posicional. El asistente no usa el decodificador generativo para inventar CVE; la respuesta es plantilla sobre documentos recuperados. La atención explica por qué el modelo puede cruzar términos aunque la pregunta no copie el documento.

### Evidencia

- Base: `paraphrase-multilingual-MiniLM-L12-v2` (12 capas, hidden 384)
- Diagrama: `docs/data/figures/fase3_arquitectura.png`

## Módulo 5 · Unidad 5 — Generativos y autosupervisados

**Archivo:** `v2 MLDS5_U5_Modelos_generativos_y_autosupervisados_v2.1 (1).pdf` (49 páginas).

### Objetivo

Modelos generativos (VAE, GAN, difusión), redes siamesas con contrastive/triplet loss, e image captioning multimodal.

### Conceptos — siamesas (núcleo de la Fase III)


- **p.18.** Redes Siamesas 2
- **p.19.** Redes Siamesas Redes Siamesas - Motivación 19 Bromley y LeCun introdujeron las redes siamesas a principios de los noventa para resolver tareas de verificación de firmas. ●Capacidad para aprender representaciones de características útiles para comparar y medir la similitud entre p…
- **p.20.** Redes Siamesas Características 20 ●Aprendizaje de similitud ●Metric Learning ●One-shot Learning ●Robustez frente al desequilibrio de clases Características Aumentar la altura del tacón Ejemplo 1 Ejemplo 2 One-shot Learning Iguales One-shot Learning Diferentes
- **p.21.** Redes Siamesas ¿Cómo funcionan? - Arquitectura 2 1 Arquitectura Función Similitud
- **p.22.** Redes Siamesas ¿Cómo funcionan? – Pares de datos 22 ●No es necesario etiquetas concretas ●Con pocas muestras se generan muchas parejas Pares de datos
- **p.23.** Redes Siamesas Contrastive Loss 23 Contrastive Loss
- **p.26.** Redes Siamesas Triplet Loss 26 El modelo toma tres entradas: ●Un ancla : Es una entrada de referencia. ●Un ejemplo positivo ●Un ejemplo negativo Entradas del Modelo
- **p.27.** Redes Siamesas Triplet Loss
- **p.28.** Redes Siamesas Aplicaciones 28 • Verificación de Firmas • Reconocimiento Facial • Rastreo de objetos • Aprendizaje con pocos ejemplos Aplicaciones Singh, P. (2021, 16 diciembre). Introduction To Siamese Networks - Analytics Vidhya - Medium. Medium Comparar Imagen B Imagen A Image…

### Conceptos — generativos (contexto, no despliegue)


- **p.7.** ¿Cómo podemos aprender la distinción de nuestros datos? Modelos Generativos Modelos Generativos Modelos de variables latentes Objetivo Existen dos formas de generar datos: aprendiendo su distribución o reduciendo gradualmente el ruido que hay en estas. Modelos de difusión Muestra…
- **p.8.** Modelos Generativos Modelos de variables latentes Autoencoders y Autoencoder variacional (VAEs) Redes generativas adversarias (GANs) Modelos de Variable Latentes
- **p.9.** Modelos Generativos Autoencoders Encoder Decoder Representación Latente Función de perdida
- **p.13.** Modelos Generativos Generative Adversarial Networks (GANs) 13 Generador de red Muestra "falsa" de la representación aprendida de la distribución de datos
- **p.15.** Modelos Generativos Modelos de difusión 15 Denoiser Función de pérdida
- **p.30.** Image Captioning 3 Un tren circulando por una vía junto a un bosque. Un grupo de chicos jugando al fútbol en un campo.
- **p.31.** Image Captioning Idea General 31 Modelo Visual Features Modelo de Texto Generativo Caption

### Implicación

El bi-encoder **es** la red siamesa del curso (p.19–27): dos torres que comparten pesos, pares ancla–positivo, pérdida contrastiva. MultipleNegativesRankingLoss / InfoNCE es la forma usada en sentence-transformers. VAE/GAN/difusión y captioning no entran al SOC. QLoRA generativo sigue fuera de este cierre.

### Evidencia

- Pares: `src/threatintel/rag/pairs.py`
- Pérdida y entrenamiento: `src/threatintel/rag/train.py`
- Decisión de no desplegar 3 épocas: `docs/modeling/hiperparametros.md`

## Tabla curso → componente del proyecto

| Unidad | Concepto del curso | Componente en el repo |
|--------|--------------------|------------------------|
| M4U2 | Preparación del corpus | hardening + `lineage_gate` |
| M4U3 | TF-IDF / embeddings | `classical.py`, baseline MiniLM |
| M4U4 | Train / val / test | pares, grilla en dev, gold hold-out |
| M4U5 | Similitud semántica | coseno + MRR / Recall@k |
| M5U1 | Fine-tuning DL | `train_biencoder` |
| M5U3 | CNN | fuera de alcance |
| M5U4 | Encoder Transformer | MiniLM 12 capas |
| M5U5 | Siamesa + contrastive | bi-encoder InfoNCE |
| M5U5 | Generativos imagen | fuera de alcance |

## Límites de la extracción

U2 y U3 del Módulo 4 y varias páginas de U1 M4 / U1 M5 son diapositivas rasterizadas: el OCR no devolvió texto. La base usa el índice de temas visible en las páginas con texto y el uso real en el código.
No está el PDF de la Unidad 2 del Módulo 5 en `Recursos/`.
Los JSON crudos por página están en `docs/knowledge/_extract/` para auditoría.
