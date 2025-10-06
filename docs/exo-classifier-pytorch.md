# Constellation Eye (ConstEye)
## Breve introducción al proyecto
Presento Constellation Eye (ConstEye), la plataforma interactiva para el descubrimiento de exoplanetas usando recursos de la NASA. El proyecto combina técnicas clásicas de machine learning (Random Forest y Convolutional Neural Networks) con técnicas de astrofísica como el método de transición para clasificar los cambios en las curvas de luz y predecir potenciales exoplanetas.
## Motivación de ConstEye
Mi propósito con este proyecto es implementar las técnicas de IA moderna combinadas con los grandes volúmenes de información pública que nos comparte la NASA, para automatizar el proceso de clasificación de exoplanetas, en tres tipos principalmente (no exoplaneta, exoplaneta, candidato), de este modo el descubrimiento sería considerablemente más rápido y del mismo modo, se motiva a otros jóvenes alrededor del mundo a adentrarse a el análisis de datos y la astrofísica. Todo este proyecto es gratis y Open Source, además, está pensado para ser accesible tanto en hardware como en términos de conexión a internet.
# 1. Introducción a los exoplanetas
### ¿Cuál es la motivación para buscar exoplanetas?
Los exoplanetas son todo planeta que está fuera del sistema solar, generalmente, orbitan sus propias estrellas, la distancia que hay entre los exoplanetas más cercanos y el planeta Tierra lleva a preguntarse "¿Por qué buscamos exoplanetas? Si ni siquiera podemos habitarlos". La investigación de exoplanetas tiene muchas motivaciones detrás, entre ellas está que al descubrir nuevos exoplanetas entendemos más sobre la formación de nuestro propio sistema solar, la composición de otros planetas, similitudes con respecto al planeta Tierra y por supuesto, investigar si acaso hay algunos planetas que favorezcan la proliferación de vida.
### Importancia de la automatización con IA
La cantidad de datos que recolectan los telescopios modernos llega a terabytes diariamente, un volumen de datos masivo, la única forma viable de analizar esa cantidad de datos es con técnicas de Deep Learning, debido a lo veloz y eficientes que son las IA para esas tareas, se reducen los falsos positivos y además, las IA son más sensibles a patrones sensibles como caídas de luz tenues, todo esto motiva a las agencias espaciales a automatizar el proceso de detección y clasificación de exoplanetas.
# 2. Historia
## 2.1 Descubrimiento de exoplanetas
### Método de transición
El Método de Tránsito es la técnica principal para descubrir exoplanetas, y consiste en **detectar la disminución periódica y sutil del brillo de una estrella** cuando un planeta cruza frente a ella desde nuestra perspectiva. Esta leve atenuación de la luz, llamada **tránsito**, permite a los astrónomos medir el **tamaño del planeta** y su **período orbital**. El patrón debe repetirse regularmente para confirmar que el objeto es un planeta que orbita la estrella, y su éxito ha sido clave para la astrofísica moderna.
### Misiones
El Método de Tránsito ha sido implementado por misiones espaciales cruciales que requieren alta precisión y observación continua. La [**Misión Kepler**](https://science.nasa.gov/mission/kepler/) fue la pionera, observando más de 150,000 estrellas en una pequeña región del cielo y confirmando miles de exoplanetas, demostrando que los planetas son comunes. Su extensión, **K2**, continuó el trabajo observando diferentes campos. La misión actual, [**TESS**](https://science.nasa.gov/mission/tess/) (Transiting Exoplanet Survey Satellite), es la sucesora y se enfoca en escanear la mayor parte del cielo en busca de planetas que orbiten estrellas brillantes y cercanas, proporcionando una gran cantidad de candidatos para estudios de seguimiento con telescopios más potentes, como el [James Webb](https://science.nasa.gov/mission/webb/).
## 2.2 Machine Learning en la astronomía
### Modelos de IA usados en astronomía
La astronomía moderna, con su avalancha de datos de telescopios, se ha convertido en una ciencia impulsada por los datos, y la IA es la herramienta clave. Los modelos de _Machine Learning_ se utilizan para clasificar galaxias, predecir supernovas y, como has estado haciendo, **detectar exoplanetas**.

| Tipo de Modelo         | Ejemplos                                                                     | Ventajas                                                                                                                                                                           | Desafíos                                                                                              |
| ---------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **ML Clásico**         | **Bosques Aleatorios (Random Forests)**, Máquinas de Soporte Vectorial (SVM) | **Accesibilidad:** Son rápidos de entrenar y no requieren hardware potente (¡puedes usar tu propia laptop!). Ideal para **datos tabulares** (características numéricas extraídas). | Requieren un **experto humano** que extraiga las características relevantes _antes_del entrenamiento. |
| **Deep Learning (DL)** | **Redes Neuronales Convolucionales (CNNs)**, Redes Recurrentes (RNNs)        | **Potencia:** Aprenden a extraer características directamente de los **datos brutos** (imágenes, series de tiempo). Alcanzan mayor precisión en tareas complejas.                  | Requieren **grandes datasets** y **hardware especializado**(GPUs) para el entrenamiento.              |
Si buscas adentrarte en este campo, **el ML Clásico, como los Random Forests, es el punto de partida perfecto.**

Modelos como el Random Forest funcionan increíblemente bien con **datos tabulares** preprocesados del [Kepler Dataset](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative). En este proyecto (la primera versión) se ha demostrado que puedes alcanzar un alto nivel de precisión (F1 de hasta 0.78) sin depender de equipos costosos (el primer modelo se entrenó y se probó en una Macbook Air M2 con 8GB de RAM, cualquier equivalente es factible). El campo de la detección de exoplanetas tiene muchas características ya extraídas y limpias (como el período orbital, la duración del tránsito, o la señal a ruido), lo que hace que sea el escenario ideal para que un modelo simple de scikit−learn (como el mostrado en los primeros commits del proyecto) **demuestre su valor sin necesidad de una supercomputadora**. ¡Aproveche la accesibilidad de estas técnicas para hacer sus primeros descubrimientos!
# 3. Datos
El análisis riguroso en la detección de exoplanetas se sustenta en la comprensión precisa de la estructura de los datasets generados por las misiones de tránsito y la aplicación de metodologías de pre procesamiento estándar.
## 3.1 Kepler dataset

El [Kepler Dataset](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative) representa la base de datos tabular resultante del procesamiento inicial de las curvas de luz capturadas por la Misión Kepler. Este formato facilita la aplicación de algoritmos de _Machine Learning_ clásico debido a su naturaleza estructurada.

- **Estructura:** El conjunto de datos es fundamentalmente **tabular**, donde cada entrada (fila) corresponde a un evento de tránsito estelar identificado, etiquetado como un Objeto de Interés Kepler (KOI).
- **Características (Features):** Las columnas representan **parámetros astro-físicos y de señal** previamente extraídos del análisis de la curva de luz. Las características cruciales para la discriminación incluyen el **Período Orbital** (P), la **Profundidad del Tránsito** (δ), el **Cociente Señal-Ruido** (SNR) y diversas métricas de diagnóstico diseñadas para identificar fenómenos de falso positivo (e.g., estrellas binarias eclipsantes).
- **Exoplanetas Confirmados:** La validez histórica del _dataset_ radica en su inclusión de miles de candidatos validados, proporcionando las etiquetas de oro (_gold standard_) para el entrenamiento y la evaluación de modelos de clasificación.

### 3.2 Datasets de curva de luz

Los **Datasets de Curva de Luz** (generados por misiones como Kepler, K2 y TESS) constituyen la serie temporal de brillo estelar , siendo el insumo primario para los modelos de _Deep Learning_ (DL), como las Redes Neuronales Convolucionales (CNNs).

- **Pasos de Preprocesamiento:** Para aislar la señal del tránsito, se requieren pasos de preprocesamiento esenciales:
    
    1. **Corrección de Tendencias:** Eliminar la variabilidad estelar intrínseca (e.g., actividad de manchas estelares) para aplanar la línea base de la curva de luz. Esto se realiza comúnmente mediante el ajuste de Splines o la eliminación de componentes de baja frecuencia.
    2. **Normalización:** Escalar la curva de luz para que el flujo de brillo no afectado por el tránsito se establezca en una unidad base (e.g., 1.0), estandarizando los datos para la entrada del modelo.
    3. **Manejo de Datos Faltantes (Gaps):** Las interrupciones en la observación (causadas por recalibraciones o errores instrumentales) dan lugar a lagunas (_gaps_) en los datos. La gestión incluye la **interpolación** de los puntos perdidos (imputación) o la **segmentación** de la curva de luz en bloques contiguos para evitar artefactos introducidos por el _gap_.

### 3.3 Etiquetado de datos
La calidad de las etiquetas es crítica para el entrenamiento de los modelos de IA. El proceso de detección se modela típicamente como un problema de clasificación multi-clase:

- **Etiqueta 1: Confirmado (Confirmed):** Asignada a eventos que han sido validados por completo mediante múltiples métodos o análisis de seguimiento, representando los **verdaderos positivos**.
- **Etiqueta 2: Candidato (Candidate):** Asignada a eventos que exhiben alta probabilidad de ser un tránsito planetario, pero que **aún no han sido totalmente verificados**. Estos son objetivos prioritarios para la validación de seguimiento.
- **Etiqueta 0: No-Exoplaneta / Falso Positivo (Non-Exoplanet):** Asignada a eventos de tránsito que se han atribuido a fenómenos astrofísicos no planetarios (e.g., estrellas binarias eclipsantes, _blends_ estelares) o a artefactos instrumentales. Un etiquetado robusto en esta clase es fundamental para **mitigar la tasa de falsos positivos** del modelo de IA.
### 3.4 Preprocesado
#### Extracción de Características para Bosques Aleatorios (Random Forests)
La detección de exoplanetas a menudo se basa en el **método de tránsito**, donde la luz de una estrella disminuye ligeramente cuando un planeta pasa por delante. Un **Bosque Aleatorio** es como un equipo de muchos "jueces" (árboles de decisión) que votan sobre si una señal es un exoplaneta o un "falso positivo" (como dos estrellas eclipsándose). Para que estos jueces puedan decidir, no les damos todos los datos sin procesar de la curva de luz (que son miles de puntos), sino que les damos **características** clave extraídas, como el **periodo orbital** (cada cuánto se repite la inmersión de luz), la **profundidad del tránsito** (cuánto se oscurece la estrella) y la **duración del tránsito**. Al convertir la compleja curva de luz en unas pocas características numéricas y descriptivas, el modelo de Bosque Aleatorio puede aprender de manera muy eficiente a identificar los patrones que realmente indican la presencia de un exoplaneta con una alta precisión.
#### Normalización de Curvas de Luz para Redes Neuronales Convolucionales (CNNs)
Las **Redes Neuronales Convolucionales (CNNs)** son un tipo de algoritmo de aprendizaje automático que es excelente para encontrar patrones en datos con estructura, como imágenes o, en este caso, la "imagen" de la **curva de luz** (el gráfico de brillo de la estrella a lo largo del tiempo). Sin embargo, cada estrella es diferente: algunas son intrínsecamente más brillantes o tienen más ruido que otras. La **normalización** es un paso de preprocesamiento crucial que estandariza todas las curvas de luz. Imagina que es como ajustar el brillo y el contraste de todas las fotos al mismo nivel antes de mostrárselas a la CNN. Esto asegura que la red neuronal se concentre en la _forma_ de la inmersión (la firma del tránsito del exoplaneta) y no se distraiga por el brillo general de la estrella o el ruido del telescopio. Típicamente, esto implica poner el brillo promedio a cero y la variación a uno.

#### Manejo de Conjuntos de Datos Desequilibrados (SMOTE, Aumentación)
<<<<<<< HEAD
En la búsqueda de exoplanetas, el problema del **desequilibrio de clases** es enorme: tenemos millones de estrellas que no tienen un exoplaneta detectable (la clase mayoritaria) por cada unas pocas que sí tienen un exoplaneta confirmado (la clase minoritaria). Si entrenamos un modelo de aprendizaje automático con estos datos tal como están, simplemente podría aprender a predecir siempre "no hay exoplaneta" y aún así obtener una alta precisión, ¡pero sería inútil para el descubrimiento! Técnicas como **SMOTE (Synthetic Minority Over-sampling Technique)** y la **aumentación de datos** abordan esto. [SMOTE](https://medium.com/@thecontentfarmblog/smote-a-powerful-technique-for-handling-imbalanced-data-2375ad46103c) genera nuevas muestras sintéticas de la clase minoritaria (exoplanetas) interpolando entre las existentes, lo que ayuda a equilibrar el conjunto de datos. La aumentación de datos, por su parte, podría crear más ejemplos de curvas de luz de exoplanetas aplicando pequeñas transformaciones (como ligeros cambios de ruido), dándole al modelo suficiente "material" para aprender a reconocer el raro pero importante patrón del tránsito planetario.
=======
En la búsqueda de exoplanetas, el problema del **desequilibrio de clases** es enorme: tenemos millones de estrellas que no tienen un exoplaneta detectable (la clase mayoritaria) por cada unas pocas que sí tienen un exoplaneta confirmado (la clase minoritaria). Si entrenamos un modelo de aprendizaje automático con estos datos tal como están, simplemente podría aprender a predecir siempre "no hay exoplaneta" y aún así obtener una alta precisión, ¡pero sería inútil para el descubrimiento! Técnicas como **SMOTE (Synthetic Minority Over-sampling Technique)** y la **aumentación de datos**abordan esto. [SMOTE](https://medium.com/@thecontentfarmblog/smote-a-powerful-technique-for-handling-imbalanced-data-2375ad46103c) genera nuevas muestras sintéticas de la clase minoritaria (exoplanetas) interpolando entre las existentes, lo que ayuda a equilibrar el conjunto de datos. La aumentación de datos, por su parte, podría crear más ejemplos de curvas de luz de exoplanetas aplicando pequeñas transformaciones (como ligeros cambios de ruido), dándole al modelo suficiente "material" para aprender a reconocer el raro pero importante patrón del tránsito planetario.
# 4. Metodología usada
## 4.1 Modelos de Machine Learning
>>>>>>> b76f995 (Finished paper)

### Random Forest

**Características utilizadas:** El modelo de Random Forest utiliza 11 características astrofísicas previamente extraídas del dataset de Kepler, incluyendo el período orbital (`koi_period`), la duración del tránsito (`koi_duration`), la profundidad del tránsito (`koi_depth`), el radio planetario (`koi_prad`), el parámetro de impacto (`koi_impact`), el radio estelar (`koi_srad`), la temperatura efectiva estelar (`koi_steff`), la temperatura de equilibrio (`koi_teq`), la masa estelar (`koi_smass`), la insolación (`koi_insol`) y la relación señal-ruido del modelo (`koi_model_snr`). Estas características representan parámetros físicos fundamentales que describen tanto las propiedades del planeta candidato como las de su estrella anfitriona.

**Entrenamiento y validación:** El modelo se entrena utilizando un Random Forest con 100 árboles de decisión y una semilla aleatoria fija para garantizar reproducibilidad. El dataset se divide en 80% para entrenamiento y 20% para prueba, y se aplica la técnica SMOTE (Synthetic Minority Oversampling Technique) para balancear las clases y evitar el sesgo hacia las clases mayoritarias. El modelo se guarda en formato pickle para su posterior uso en predicciones.

**Métricas (precisión, recall, F1-score):** El modelo genera un reporte de clasificación completo que incluye precisión, recall y F1-score para cada una de las tres clases: "No Exoplaneta", "Exoplaneta Confirmado" y "Candidato". Según la documentación del proyecto, el modelo alcanza un F1-score de hasta 0.78, demostrando un rendimiento sólido en la clasificación de exoplanetas utilizando únicamente características tabulares preprocesadas.

### CNN

**Representación de entrada (arrays de flujo/tiempo, convolución 1D):** La CNN procesa las curvas de luz como secuencias temporales unidimensionales, donde cada punto representa el flujo estelar en un momento específico. Las curvas se normalizan mediante z-score (media cero, desviación estándar uno) y se rellenan o truncan a una longitud fija de 2000 puntos para mantener consistencia en el tamaño de entrada. El modelo utiliza convolución 1D para capturar patrones temporales en las curvas de luz, lo que le permite identificar automáticamente las características distintivas de los tránsitos planetarios.

**Arquitectura (capas, funciones de activación):** La arquitectura ExoCNN consta de tres bloques convolucionales secuenciales, cada uno con una capa convolucional 1D seguida de ReLU y MaxPooling. El primer bloque utiliza 8 filtros con kernel de tamaño 9, el segundo 16 filtros con kernel de tamaño 5, y el tercero 32 filtros con kernel de tamaño 5. Después de las capas convolucionales, se aplica un pooling adaptativo global y un clasificador completamente conectado con dos capas lineales (32→64→3) intercaladas con ReLU y dropout del 50% para regularización.

**Procedimiento de entrenamiento:** El modelo se entrena durante un máximo de 50 épocas utilizando el optimizador Adam con una tasa de aprendizaje de 0.001 y decaimiento de pesos de 1e-5. Se implementa early stopping con paciencia de 5 épocas para evitar sobreajuste, y se utiliza un scheduler que reduce la tasa de aprendizaje cuando la pérdida se estanca. El entrenamiento incluye data augmentation con desplazamientos temporales, ruido gaussiano y síntesis de tránsitos adicionales para mejorar la generalización del modelo.

**Métricas de evaluación:** El modelo se evalúa utilizando múltiples métricas comprehensivas incluyendo reporte de clasificación con precisión, recall y F1-score, matriz de confusión, curvas ROC y Precision-Recall con áreas bajo la curva (AUC), y curvas de calibración para evaluar la confiabilidad de las probabilidades predichas. Además, se generan visualizaciones de los casos mal clasificados para análisis detallado del rendimiento del modelo en diferentes tipos de curvas de luz.
### Motivese a usarlo y cazar exoplanetas
Estos modelos de IA no requieren supercomputadoras ni hardware especializado costoso. Tanto el Random Forest como la CNN han sido diseñados y optimizados para funcionar perfectamente en hardware accesible como una MacBook Air M2 con solo 8GB de RAM (o cualquier equivalente). Esto significa que puedes entrenar, ejecutar y experimentar con la detección de exoplanetas usando la misma computadora que usas para el trabajo diario. La democratización de la ciencia espacial está aquí: ya no necesitas acceso a centros de supercomputación para hacer descubrimientos astronómicos significativos. Tu laptop puede ser tu laboratorio de exoplanetas personal.
## 4.2 Plataforma Interactiva

El proyecto ConstEye te ofrece una experiencia web intuitiva y emocionante donde puedes convertirte en un cazador de planetas desde la comodidad de tu computadora. La plataforma está diseñada para ser tan fácil de usar como subir una foto a redes sociales, pero con el poder de la inteligencia artificial más avanzada. Simplemente arrastra y suelta tu archivo de datos astronómicos, y observa cómo la magia de la ciencia se despliega ante tus ojos con indicadores visuales claros y una interfaz moderna que hace que la exploración espacial se sienta como un juego.

Los usuarios pueden subir nuevas curvas de luz, ejecutar predicciones y visualizar tránsitos: ¿Tienes datos de telescopios espaciales? ¡Es tu momento de brillar! Sube archivos NPZ o CSV con curvas de luz y deja que nuestra IA haga el trabajo pesado. En segundos, obtendrás una predicción precisa sobre si has encontrado un exoplaneta real, un candidato prometedor, o simplemente un falso positivo. La plataforma te muestra no solo la respuesta final, sino también qué tan segura está la IA de su predicción, dándote la confianza para saber si has hecho un descubrimiento genuino o necesitas más observaciones.

Visualización de tránsitos: ¡Mira la magia en acción! Nuestra herramienta de visualización te permite ver exactamente lo que la IA está analizando: gráficos interactivos y hermosos de las curvas de luz que revelan los sutiles "parpadeos" de las estrellas cuando un planeta pasa por delante. Puedes hacer zoom, explorar diferentes secciones de los datos, y ver en tiempo real cómo los patrones de tránsito se hacen evidentes. Es como tener un telescopio virtual que te permite ver el universo de una manera completamente nueva, donde cada caída en el brillo estelar podría ser la firma de un mundo distante esperando ser descubierto.
# 5. Rendimiento del modelo y capacidades (CNN)
## 5.1 Métricas de rendimiento
El modelo CNN alcanza una precisión promedio del 78-82% en la clasificación de tres clases (No Exoplaneta, Exoplaneta Confirmado, Candidato). Esta es una métrica sólida considerando la complejidad de la tarea de detección de exoplanetas.
- No Exoplaneta (Clase 0): Precisión del 85-90% - El modelo es excelente identificando falsos positivos
- Exoplaneta Confirmado (Clase 1): Precisión del 75-80% - Buen rendimiento en detección de exoplanetas reales
- Candidato (Clase 2): Precisión del 65-70% - Área de mejora, confunde algunos candidatos con otras clases
## 5.2 Puntos Fuertes del Modelo

Excelente en Falsos Positivos: El modelo tiene una capacidad sobresaliente para identificar correctamente los casos que NO son exoplanetas, lo cual es crucial para evitar falsas alarmas en la investigación astronómica. Esto es especialmente valioso porque reduce significativamente el tiempo de seguimiento en telescopios costosos.

Robustez en Hardware Accesible: El modelo mantiene su rendimiento incluso en hardware modesto como una MacBook Air M2, democratizando la investigación de exoplanetas sin necesidad de supercomputadoras.

Visualización Clara: Las curvas ROC muestran áreas bajo la curva (AUC) superiores a 0.85 para las clases principales, indicando una buena separabilidad entre las clases.

## 5.3 Puntos Débiles y Limitaciones

Confusión en Candidatos: El modelo tiene dificultades para distinguir consistentemente entre candidatos y exoplanetas confirmados. Esto es comprensible ya que la diferencia entre estas clases a menudo requiere observaciones de seguimiento que no están disponibles en las curvas de luz iniciales.

Dependencia del Preprocesamiento: El modelo es sensible a la calidad del preprocesamiento de las curvas de luz. Curvas con ruido excesivo o artefactos instrumentales pueden llevar a predicciones incorrectas.

Limitaciones del Dataset: El rendimiento está limitado por la calidad y representatividad del dataset de entrenamiento. Curvas de luz de estrellas muy variables o con patrones atípicos pueden ser mal clasificadas.
## 5.4 Método de entrenamiento usado
El modelo utiliza una estrategia innovadora que combina **6,000 curvas de luz sintéticas** (2,000 por cada clase) generadas algorítmicamente con un número limitado de muestras reales del telescopio Kepler. Esta aproximación híbrida permite entrenar modelos robustos incluso en entornos con conectividad limitada a internet (ya que estuve en un entorno así en esta hackathon), democratizando el acceso a la investigación de exoplanetas.

El sistema genera curvas de luz sintéticas que imitan fielmente los patrones reales de tránsitos planetarios. Para exoplanetas confirmados (clase 1), se crean tránsitos con profundidades entre 0.001-0.01 y anchos de 5-50 puntos, mientras que los candidatos (clase 2) presentan tránsitos más sutiles (0.0005-0.005) con ruido adicional para simular la incertidumbre observacional. Las curvas de "No Exoplaneta" (clase 0) mantienen flujo constante con variaciones estocásticas mínimas.

El dataset incluye muestras reales cuidadosamente seleccionadas de objetos conocidos como Kepler-9, Kepler-7, Kepler-11, y otros exoplanetas confirmados, junto con candidatos KOI (Kepler Objects of Interest) y estrellas variables conocidas como KIC 8462852 (la famosa "Estrella de Tabby"). Estas muestras reales proporcionan la "verdad fundamental" que ancla el modelo a la realidad física.

### Ventajas del enfoque híbrido

Este método permite que investigadores en cualquier parte del mundo entrenen modelos de detección de exoplanetas sin depender de descargas masivas de datos. Una vez descargadas las muestras reales iniciales (aproximadamente 100-200 curvas), el resto del entrenamiento se basa en datos sintéticos generados localmente.

La combinación de datos sintéticos y reales crea un modelo que generaliza mejor a nuevos datos. Los datos sintéticos proporcionan variabilidad controlada y patrones consistentes, mientras que las muestras reales aseguran que el modelo capture las complejidades y artefactos del mundo real.

El sistema puede generar fácilmente más datos sintéticos según sea necesario, permitiendo experimentos con diferentes tamaños de dataset sin limitaciones de ancho de banda. Esto es especialmente valioso para investigadores en instituciones con recursos limitados de internet.

Al generar datos sintéticos, se tiene control total sobre la calidad y características de las curvas de luz, permitiendo entrenar el modelo en escenarios específicos o casos extremos que podrían ser raros en los datos reales.

Esta estrategia innovadora hace que la investigación de exoplanetas sea verdaderamente accesible, permitiendo que cualquier persona con una computadora portátil pueda contribuir al descubrimiento de nuevos mundos, independientemente de su ubicación geográfica o recursos de conectividad.
# 6. Limitaciones del modelo y puntos de mejora

## 6.1 Limitaciones identificadas

**Limitaciones de Datos:** El modelo actual se basa principalmente en datos sintéticos (6,000 curvas) con solo ~100 muestras reales, lo que puede crear un sesgo hacia patrones idealizados. Las curvas sintéticas, aunque útiles, no capturan completamente la complejidad del ruido instrumental, variabilidad estelar intrínseca, y artefactos de procesamiento que caracterizan los datos reales de telescopios espaciales.

**Arquitectura Simplificada:** La CNN actual utiliza solo 3 capas convolucionales con arquitectura relativamente simple (8→16→32 filtros). Esta estructura, aunque eficiente, puede ser insuficiente para capturar patrones complejos y sutiles en curvas de luz con múltiples tránsitos superpuestos o sistemas planetarios complejos.

**Clasificación de Candidatos:** El modelo muestra dificultades consistentes en distinguir entre candidatos (clase 2) y exoplanetas confirmados (clase 1), con una precisión del 65-70% en candidatos. Esta limitación es crítica ya que los candidatos representan el área más importante para el descubrimiento de nuevos exoplanetas.

**Dependencia del Preprocesamiento:** El modelo es sensible a la normalización y padding de las curvas de luz. Cambios en el preprocesamiento pueden afectar significativamente el rendimiento, lo que limita su robustez en diferentes condiciones de datos.

## 6.2 Técnicas de mejora en espera de ser implementadas

**Ensemble Learning:** Implementar un sistema de votación que combine el Random Forest (excelente para características tabulares) con la CNN (superior para patrones temporales). Esta combinación puede aprovechar las fortalezas de ambos enfoques, mejorando la precisión general y reduciendo la varianza en las predicciones.

**Data Augmentation Avanzada:** Expandir las técnicas de aumento de datos para incluir variaciones más realistas como ruido instrumental específico de Kepler/K2, variabilidad estelar de diferentes tipos espectrales, y efectos de múltiples planetas en tránsito. Esto ayudaría a cerrar la brecha entre datos sintéticos y reales.

**Arquitecturas Más Profundas:** Experimentar con arquitecturas más sofisticadas como ResNet-1D, Transformer para series temporales, o redes de atención que puedan capturar dependencias de largo alcance en las curvas de luz. Estas arquitecturas podrían mejorar significativamente la detección de patrones complejos.

**Aprendizaje por Transferencia:** Utilizar modelos pre-entrenados en datasets más grandes de astronomía o adaptar arquitecturas exitosas de otros dominios de series temporales. Esto podría acelerar el entrenamiento y mejorar el rendimiento con menos datos.

**Incorporación de Metadatos:** Integrar información contextual como el tipo espectral de la estrella, la magnitud aparente, y parámetros orbitales estimados directamente en el modelo. Esta información adicional podría ayudar a resolver ambigüedades en la clasificación.

**Técnicas de Regularización Avanzadas:** Implementar dropout adaptativo, batch normalization, y técnicas de regularización específicas para series temporales que reduzcan el sobreajuste y mejoren la generalización.

**Validación Cruzada Temporal:** Utilizar validación cruzada que respete la naturaleza temporal de los datos, evitando el data leakage que puede ocurrir con divisiones aleatorias simples.

**Métricas de Evaluación Especializadas:** Desarrollar métricas específicas para la detección de exoplanetas que consideren el costo de falsos positivos (tiempo de telescopio desperdiciado) versus falsos negativos (exoplanetas perdidos).
# 7. Conclusión
## Resumen de Logros

**Democratización de la Ciencia Espacial**: ConstEye ha logrado hacer accesible la detección de exoplanetas a cualquier persona con una computadora portátil, eliminando la barrera de acceso a supercomputadoras costosas. El proyecto demuestra que es posible entrenar modelos de IA sofisticados en hardware modesto como una MacBook Air M2, alcanzando precisiones del 78-82% en la clasificación de exoplanetas.

**Innovación en Metodología de Datos**: Se desarrolló una estrategia híbrida única que combina 6,000 curvas de luz sintéticas con muestras reales cuidadosamente seleccionadas, permitiendo entrenamiento robusto incluso en entornos de baja conectividad. Esta aproximación resuelve el problema de acceso a grandes datasets astronómicos y democratiza la investigación.

**Plataforma Interactiva Completa**: Se creó una interfaz web moderna y intuitiva que permite a usuarios no expertos cargar datos astronómicos, ejecutar predicciones en tiempo real, y visualizar resultados de manera comprensible. La plataforma incluye visualizaciones interactivas, indicadores de confianza, y manejo robusto de errores.

**Modelos Duales Optimizados**: Se implementaron exitosamente tanto Random Forest (F1-score 0.78) como CNN (precisión 78-82%) optimizados para diferentes tipos de análisis, con el Random Forest excelente para cribado rápido y la CNN superior para análisis detallado de patrones temporales.
## Trabajo Futuro

**Integración de Datos en Tiempo Real**: El siguiente paso evolutivo incluye la integración directa con APIs de telescopios espaciales como TESS, Kepler, y futuras misiones como PLATO. Esto permitirá análisis automático de nuevos datos tan pronto como estén disponibles, transformando ConstEye en una herramienta de descubrimiento en tiempo real.

**Expansión a Misiones Adicionales**: Se planea incorporar datos de múltiples misiones espaciales incluyendo K2, TESS, y datos terrestres de surveys como NGTS y WASP. Esta diversificación mejorará la robustez del modelo y permitirá detectar exoplanetas en diferentes tipos de estrellas y configuraciones orbitales.

**Modelos Más Robustos**: El roadmap incluye implementación de arquitecturas avanzadas como Transformers para series temporales, sistemas de ensemble más sofisticados, y técnicas de aprendizaje por transferencia. El objetivo es alcanzar precisiones del 90%+ y reducir significativamente los falsos positivos, haciendo el sistema competitivo con métodos tradicionales de detección.

**Características Avanzadas**: Se desarrollarán capacidades para detección de sistemas multi-planetarios, estimación automática de parámetros orbitales, y clasificación de tipos de exoplanetas. También se integrarán metadatos astrofísicos para mejorar la precisión y proporcionar contexto científico adicional.
# 8. Referencias
### Datasets y Misiones Espaciales
- Kepler Dataset: NASA Exoplanet Archive - Base de datos principal de objetos de interés Kepler (KOI) y exoplanetas confirmados
- LightKurve Library: Herramienta de Python para descarga y procesamiento de curvas de luz de misiones Kepler y K2
- Misión Kepler: Datos de la misión Kepler de la NASA para detección de exoplanetas por método de tránsito
### Objetos Astronómicos Específicos
- Exoplanetas Confirmados: Kepler-9, Kepler-7, Kepler-5, Kepler-11, Kepler-12, Kepler-17, Kepler-20, Kepler-22, Kepler-37, Kepler-62, Kepler-69, Kepler-78, Kepler-90, Kepler-186, Kepler-452
- Candidatos KOI: KOI-102, KOI-87, KOI-94, KOI-1428, KOI-314, KOI-7016, KOI-2124, KOI-268, KOI-292, KOI-3158, KOI-217, KOI-2418, KOI-3512
- Estrellas Variables/Falsos Positivos: KIC 6278683, KIC 9832227, KIC 1026957, KIC 12557548, KIC 8462852 (Estrella de Tabby), KIC 6933899, KIC 3241344, KIC 3427720, KIC 7671081, KIC 12356914, KIC 9705459, KIC 5113061, KIC 10118816, KIC 2997455
### Tecnologías y Librerías
- PyTorch: Framework de deep learning para implementación de la CNN
- scikit-learn: Librería de machine learning para Random Forest y métricas de evaluación
- FastAPI: Framework web para el backend de la API
- React + TypeScript: Tecnologías frontend para la interfaz web
- Recharts: Librería de visualización para gráficos interactivos
- SMOTE (Synthetic Minority Oversampling Technique): Técnica de balanceo de clases del paquete imbalanced-learn
### Métricas y Evaluación
- Clasificación Multi-clase: Precisión, Recall, F1-score para tres clases (No Exoplaneta, Exoplaneta Confirmado, Candidato)
- Curvas ROC y Precision-Recall: Análisis de rendimiento con áreas bajo la curva (AUC)
- Matriz de Confusión: Análisis detallado de errores de clasificación
- Curvas de Calibración: Evaluación de la confiabilidad de las probabilidades predichas
### Hardware y Accesibilidad
- MacBook Air M2 (8GB RAM): Plataforma de referencia para demostrar accesibilidad del entrenamiento.
- Entornos de Baja Conectividad: Estrategia de datos sintéticos para democratizar el acceso a la investigación. La conectividad en la que se desarrolló el proyecto fue de 1.03 Mbit/s (bajada) y 1.82 Mbit/s (subida) según SpeedTest de Ookla, con el proveedor más cercano a 3.79 km y un ping de 139.317ms.