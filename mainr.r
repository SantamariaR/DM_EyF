# limpio la memoria
Sys.time()
rm(list=ls(all.names=TRUE)) # remove all objects
gc(full=TRUE, verbose=FALSE) # garbage collection

PARAM <- list()
PARAM$experimento <- "exp40"
PARAM$semilla_primigenia <- 100129

BUCKET_NAME <- "/home/rami_santamaria92/buckets/b1"

Sys.time()

# Instalar data.table si no está disponible
paquetes_necesarios <- c("data.table", "R.utils", "ggplot2", "readr", "dplyr", "zoo")

for (paquete in paquetes_necesarios) {
  if (!require(paquete, character.only = TRUE, quietly = TRUE)) {
    install.packages(paquete, repos = "https://cloud.r-project.org")
    require(paquete, character.only = TRUE)
  }
}


Sys.time()
require( "data.table" )

# leo el dataset
cat("=== INICIO: Leyendo dataset ===\n")
print(Sys.time())
flush.console()
dataset <- fread("/home/rami_santamaria92/datasets/competencia_02_crudo.csv.gz" )
cat("=== FIN: Dataset leído exitosamente ===\n")
print(Sys.time())
cat("Dimensiones del dataset:", nrow(dataset), "filas x", ncol(dataset), "columnas\n")
cat("\nPrimeras filas del dataset:\n")
print(head(dataset))
flush.console()

# calculo el periodo0 consecutivo
dsimple <- dataset[, list(
  "pos" = .I,
  numero_de_cliente,
  periodo0 = as.integer(foto_mes/100)*12 +  foto_mes%%100 )
]


# ordeno
setorder( dsimple, numero_de_cliente, periodo0 )

# calculo topes
periodo_ultimo <- dsimple[, max(periodo0) ]
periodo_anteultimo <- periodo_ultimo - 1


# calculo los leads de orden 1 y 2
dsimple[, c("periodo1", "periodo2") :=
  shift(periodo0, n=1:2, fill=NA, type="lead"),  numero_de_cliente
]

# assign most common class values = "CONTINUA"
dsimple[ periodo0 < periodo_anteultimo, clase_ternaria := "CONTINUA" ]

# calculo BAJA+1
dsimple[ periodo0 < periodo_ultimo &
  ( is.na(periodo1) | periodo0 + 1 < periodo1 ),
  clase_ternaria := "BAJA+1"
]

# calculo BAJA+2
dsimple[ periodo0 < periodo_anteultimo & (periodo0+1 == periodo1 )
  & ( is.na(periodo2) | periodo0 + 2 < periodo2 ),
  clase_ternaria := "BAJA+2"
]

# pego el resultado en el dataset original y grabo
setorder( dsimple, pos )
dataset[, clase_ternaria := dsimple$clase_ternaria ]

rm(dsimple)
gc()
Sys.time()

# Transponer el dataset para mostrar foto_mes y categorías de clase_ternaria como columnas
cat("\n=== Resumen de clase_ternaria por foto_mes ===\n")
dt_summary <- dataset[, .N, list(foto_mes, clase_ternaria)]
dt_transposed <- dcast(dt_summary, foto_mes ~ clase_ternaria, value.var = "N", fill = 0)
cols_order <- c("foto_mes", "CONTINUA", "BAJA+1", "BAJA+2", "NA")
cols_available <- intersect(cols_order, names(dt_transposed))
setcolorder(dt_transposed, cols_available)
setorder(dt_transposed, foto_mes)
print(dt_transposed)
flush.console()

setwd(BUCKET_NAME+"/exp")
experimento_folder <- PARAM$experimento
dir.create(experimento_folder, showWarnings=FALSE)
setwd( paste0(experimento_folder))

# --- Reparación de atributos dañados (todos 0s en un mes) ---
# Se deben reparar los atributos del dataset que para un cierto mes TODOS sus valores son cero.
# Solución: Reemplazar esos valores dañados por NA

# --- Definir columnas a excluir ---
columnas_excluir <- c("numero_de_cliente", "foto_mes", "clase_ternaria")

# --- Identificar columnas numéricas ---
columnas_numericas <- setdiff(
  names(dataset)[sapply(dataset, is.numeric)],
  columnas_excluir
)

# --- Reemplazar columnas con todos ceros por NA dentro de cada foto_mes ---
cat("=== Iniciando conversión de columnas con todos 0s a NA ===\n")

for (mes in sort(unique(dataset$foto_mes))) {
  
  # Filtrar subset del mes
  subset_mes <- dataset[foto_mes == mes, ..columnas_numericas]
  
  # Identificar columnas con suma total 0 (ignorando NAs)
  cols_todo_cero <- names(which(colSums(subset_mes, na.rm = TRUE) == 0))
  
  if (length(cols_todo_cero) > 0) {
    # Reemplazar 0s por NA solo en esas columnas
    for (col in cols_todo_cero) {
      dataset[foto_mes == mes & get(col) == 0, (col) := NA]
    }
    
    # --- Log: mostrar variables modificadas para ese foto_mes ---
    cat(sprintf(
      "[foto_mes = %s] columnas modificadas: %s\n",
      mes,
      paste(cols_todo_cero, collapse = ", ")
    ))
  } else {
    cat(sprintf("[foto_mes = %s] sin columnas con todos 0s\n", mes))
  }
}

cat("=== Conversión de 0s a NA completada ===\n")
cat("Shape final del dataset:", nrow(dataset), "filas x", ncol(dataset), "columnas\n")

#dataset[, mprestamos_personales := NULL ]
#dataset[, cprestamos_personales := NULL ]

Sys.time()
# Feature Engineering Intra-Mes
cat("\n=== INICIO: Feature Engineering Intra-Mes ===\n")
print(Sys.time())
flush.console()

# el mes 1,2, ..12 , podria servir para detectar estacionalidad
cat("Creando variable kmes...\n")
flush.console()
dataset[, kmes := foto_mes %% 100]

# creo un ctr_quarter que tenga en cuenta cuando
# los clientes hace 3 menos meses que estan
# ya que seria injusto considerar las transacciones medidas en menor tiempo
cat("Creando variable ctrx_quarter_normalizado...\n")
flush.console()
dataset[, ctrx_quarter_normalizado := as.numeric(ctrx_quarter) ]
dataset[cliente_antiguedad == 1, ctrx_quarter_normalizado := ctrx_quarter * 5.0]
dataset[cliente_antiguedad == 2, ctrx_quarter_normalizado := ctrx_quarter * 2.0]
dataset[cliente_antiguedad == 3, ctrx_quarter_normalizado := ctrx_quarter * 1.2]

# variable extraida de una tesis de maestria de Irlanda, se perdió el link
cat("Creando variable mpayroll_sobre_edad...\n")
flush.console()
dataset[, mpayroll_sobre_edad := mpayroll / cliente_edad]

cat("=== FIN: Feature Engineering Intra-Mes ===\n")
print(Sys.time())
flush.console()

# Feature Engineering Historico
cat("\n=== INICIO: Feature Engineering Historico ===\n")
print(Sys.time())
flush.console()

cat("Cargando librería Rcpp para funciones optimizadas...\n")
flush.console()
if( !require("Rcpp")) install.packages("Rcpp", repos = "http://cran.us.r-project.org")
require("Rcpp")

# se calculan para los 6 meses previos el minimo, maximo y
#  tendencia calculada con cuadrados minimos
# la formula de calculo de la tendencia puede verse en
#  https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Book%3A_Introductory_Statistics_(Shafer_and_Zhang)/10%3A_Correlation_and_Regression/10.04%3A_The_Least_Squares_Regression_Line
# para la maxíma velocidad esta funcion esta escrita en lenguaje C,
# y no en la porqueria de R o Python

cat("Compilando función C++ para cálculo de tendencias históricas...\n")
flush.console()

cppFunction("NumericVector fhistC(NumericVector pcolumna, IntegerVector pdesde )
{
  /* Aqui se cargan los valores para la regresion */
  double  x[100] ;
  double  y[100] ;

  int n = pcolumna.size();
  NumericVector out( 5*n );

  for(int i = 0; i < n; i++)
  {
    //lag
    if( pdesde[i]-1 < i )  out[ i + 4*n ]  =  pcolumna[i-1] ;
    else                   out[ i + 4*n ]  =  NA_REAL ;


    int  libre    = 0 ;
    int  xvalor   = 1 ;

    for( int j= pdesde[i]-1;  j<=i; j++ )
    {
       double a = pcolumna[j] ;

       if( !R_IsNA( a ) )
       {
          y[ libre ]= a ;
          x[ libre ]= xvalor ;
          libre++ ;
       }

       xvalor++ ;
    }

    /* Si hay al menos dos valores */
    if( libre > 1 )
    {
      double  xsum  = x[0] ;
      double  ysum  = y[0] ;
      double  xysum = xsum * ysum ;
      double  xxsum = xsum * xsum ;
      double  vmin  = y[0] ;
      double  vmax  = y[0] ;

      for( int h=1; h<libre; h++)
      {
        xsum  += x[h] ;
        ysum  += y[h] ;
        xysum += x[h]*y[h] ;
        xxsum += x[h]*x[h] ;

        if( y[h] < vmin )  vmin = y[h] ;
        if( y[h] > vmax )  vmax = y[h] ;
      }

      out[ i ]  =  (libre*xysum - xsum*ysum)/(libre*xxsum -xsum*xsum) ;
      out[ i + n ]    =  vmin ;
      out[ i + 2*n ]  =  vmax ;
      out[ i + 3*n ]  =  ysum / libre ;
    }
    else
    {
      out[ i       ]  =  NA_REAL ;
      out[ i + n   ]  =  NA_REAL ;
      out[ i + 2*n ]  =  NA_REAL ;
      out[ i + 3*n ]  =  NA_REAL ;
    }
  }

  return  out;
}")

cat("Función C++ compilada exitosamente.\n")
flush.console()

# calcula la tendencia de las variables cols de los ultimos 6 meses
# la tendencia es la pendiente de la recta que ajusta por cuadrados minimos
# La funcionalidad de ratioavg es autoria de  Daiana Sparta,  UAustral  2021

TendenciaYmuchomas <- function(
    dataset, cols, ventana = 6, tendencia = TRUE,
    minimo = TRUE, maximo = TRUE, promedio = TRUE,
    ratioavg = FALSE, ratiomax = FALSE) {
  gc(verbose= FALSE)
  # Esta es la cantidad de meses que utilizo para la historia
  ventana_regresion <- ventana

  last <- nrow(dataset)

  # creo el vector_desde que indica cada ventana
  # de esta forma se acelera el procesamiento ya que lo hago una sola vez
  vector_ids <- dataset[ , numero_de_cliente ]

  vector_desde <- seq(
    -ventana_regresion + 2,
    nrow(dataset) - ventana_regresion + 1
  )

  vector_desde[1:ventana_regresion] <- 1

  for (i in 2:last) {
    if (vector_ids[i - 1] != vector_ids[i]) {
      vector_desde[i] <- i
    }
  }
  for (i in 2:last) {
    if (vector_desde[i] < vector_desde[i - 1]) {
      vector_desde[i] <- vector_desde[i - 1]
    }
  }

  for (campo in cols) {
    nueva_col <- fhistC(dataset[, get(campo)], vector_desde)

    if (tendencia) {
      dataset[, paste0(campo, "_tend", ventana) :=
        nueva_col[(0 * last + 1):(1 * last)]]
    }

    if (minimo) {
      dataset[, paste0(campo, "_min", ventana) :=
        nueva_col[(1 * last + 1):(2 * last)]]
    }

    if (maximo) {
      dataset[, paste0(campo, "_max", ventana) :=
        nueva_col[(2 * last + 1):(3 * last)]]
    }

    if (promedio) {
      dataset[, paste0(campo, "_avg", ventana) :=
        nueva_col[(3 * last + 1):(4 * last)]]
    }

    if (ratioavg) {
      dataset[, paste0(campo, "_ratioavg", ventana) :=
        get(campo) / nueva_col[(3 * last + 1):(4 * last)]]
    }

    if (ratiomax) {
      dataset[, paste0(campo, "_ratiomax", ventana) :=
        get(campo) / nueva_col[(2 * last + 1):(3 * last)]]
    }
  }
}

# Creacion de LAGs
cat("\n=== Creación de variables LAG ===\n")
flush.console()

cat("Ordenando dataset por cliente y fecha...\n")
flush.console()
setorder(dataset, numero_de_cliente, foto_mes)

# todo es lagueable, menos la primary key y la clase
cols_lagueables <- copy( setdiff(
  colnames(dataset),
  c("numero_de_cliente", "foto_mes", "clase_ternaria")
))

cat("Cantidad de columnas lagueables:", length(cols_lagueables), "\n")
flush.console()

# https://rdrr.io/cran/data.table/man/shift.html

# lags de orden 1
cat("Creando lags de orden 1...\n")
flush.console()
dataset[,
  paste0(cols_lagueables, "_lag1") := shift(.SD, 1, NA, "lag"),
  by= numero_de_cliente,
  .SDcols= cols_lagueables
]

# lags de orden 2
cat("Creando lags de orden 2...\n")
flush.console()
dataset[,
  paste0(cols_lagueables, "_lag2") := shift(.SD, 2, NA, "lag"),
  by= numero_de_cliente,
  .SDcols= cols_lagueables
]

# agrego los delta lags
cat("Creando delta lags (diferencias entre períodos)...\n")
flush.console()
for (vcol in cols_lagueables)
{
  dataset[, paste0(vcol, "_delta1") := get(vcol) - get(paste0(vcol, "_lag1"))]
  dataset[, paste0(vcol, "_delta2") := get(vcol) - get(paste0(vcol, "_lag2"))]
}

cat("Variables LAG creadas exitosamente.\n")
flush.console()

# parametros de Feature Engineering Historico de Tendencias
cat("\n=== Configurando parámetros de Tendencias ===\n")
flush.console()

PARAM$FE_hist$Tendencias$run <- TRUE
PARAM$FE_hist$Tendencias$ventana <- 6
PARAM$FE_hist$Tendencias$tendencia <- TRUE
PARAM$FE_hist$Tendencias$minimo <- FALSE
PARAM$FE_hist$Tendencias$maximo <- FALSE
PARAM$FE_hist$Tendencias$promedio <- FALSE
PARAM$FE_hist$Tendencias$ratioavg <- FALSE
PARAM$FE_hist$Tendencias$ratiomax <- FALSE

cat("Parámetros configurados - Ventana:", PARAM$FE_hist$Tendencias$ventana, "meses\n")
flush.console()

# aqui se agregan las tendencias de los ultimos 6 meses

cols_lagueables <- intersect(cols_lagueables, colnames(dataset))
setorder(dataset, numero_de_cliente, foto_mes)

if( PARAM$FE_hist$Tendencias$run) {
    cat("Calculando tendencias históricas para", length(cols_lagueables), "columnas...\n")
    flush.console()
    
    TendenciaYmuchomas(dataset,
    cols = cols_lagueables,
    ventana = PARAM$FE_hist$Tendencias$ventana, # 6 meses de historia
    tendencia = PARAM$FE_hist$Tendencias$tendencia,
    minimo = PARAM$FE_hist$Tendencias$minimo,
    maximo = PARAM$FE_hist$Tendencias$maximo,
    promedio = PARAM$FE_hist$Tendencias$promedio,
    ratioavg = PARAM$FE_hist$Tendencias$ratioavg,
    ratiomax = PARAM$FE_hist$Tendencias$ratiomax
  )
  
  cat("Tendencias históricas calculadas.\n")
  flush.console()
}

cat("Total de columnas después de FE Historico:", ncol(dataset), "\n")
flush.console()

ncol(dataset)
Sys.time()

cat("=== FIN: Feature Engineering Historico completado ===\n")
cat("Total de columnas en dataset:", ncol(dataset), "\n")
print(Sys.time())
flush.console()


### Feature Engineering a partir de hojas de Random Forest
cat("\n=== INICIO: Feature Engineering con Random Forest ===\n")
print(Sys.time())
flush.console()

if( !require("lightgbm")) install.packages("lightgbm")
require("lightgbm")

AgregaVarRandomForest <- function() {

  cat( "inicio AgregaVarRandomForest()\n")
  flush.console()
  gc(verbose= FALSE)
  
  cat("Creando variable clase01 binaria...\n")
  flush.console()
  dataset[, clase01 := 0L ]
  dataset[ clase_ternaria %in% c( "BAJA+2", "BAJA+1"),
      clase01 := 1L ]
  
  cat("Clase01 creada. Positivos:", dataset[clase01 == 1L, .N], "\n")
  flush.console()

  campos_buenos <- setdiff(
    colnames(dataset),
    c( "clase_ternaria", "clase01")
  )
  
  cat("Campos para entrenamiento:", length(campos_buenos), "\n")
  flush.console()

  dataset[, entrenamiento :=
    as.integer( foto_mes %in% PARAM$FE_rf$train$training )]
  
  cat("Registros de entrenamiento:", dataset[entrenamiento == TRUE, .N], "\n")
  flush.console()

  cat("Creando lgb.Dataset...\n")
  flush.console()
  dtrain <- lgb.Dataset(
    data = data.matrix(dataset[entrenamiento == TRUE, campos_buenos, with = FALSE]),
    label = dataset[entrenamiento == TRUE, clase01],
    free_raw_data = FALSE
  )

  cat("Entrenando modelo Random Forest...\n")
  flush.console()
  modelo <- lgb.train(
     data = dtrain,
     param = PARAM$FE_rf$lgb_param,
     verbose = -100
  )

  cat( "Fin construccion RandomForest\n" )
  flush.console()
  # grabo el modelo, achivo .model
  #lgb.save(modelo, file=file.path(PARAM$experimento, "modelo.model"))
  #cat("Modelo guardado en modelo.model\n")
  #flush.console()

  qarbolitos <- copy(PARAM$FE_rf$lgb_param$num_iterations)
  cat("Cantidad de árboles:", qarbolitos, "\n")
  flush.console()

  periodos <- dataset[ , unique( foto_mes ) ]
  cat("Procesando", length(periodos), "periodos...\n")
  flush.console()

  for( periodo in  periodos )
  {
    cat( "periodo = ", periodo, "\n" )
    flush.console()
    datamatrix <- data.matrix(dataset[ foto_mes== periodo, campos_buenos, with = FALSE])

    cat( "Inicio prediccion\n" )
    flush.console()
    prediccion <- predict(
        modelo,
        datamatrix,
        type = "leaf"
    )
    cat( "Fin prediccion\n" )
    flush.console()

    for( arbolito in 1:qarbolitos )
    {
       cat( arbolito, " " )
       hojas_arbol <- unique(prediccion[ , arbolito])

       for (pos in 1:length(hojas_arbol)) {
         # el numero de nodo de la hoja, estan salteados
         nodo_id <- hojas_arbol[pos]
         dataset[ foto_mes== periodo, paste0(
            "rf_", sprintf("%03d", arbolito),
             "_", sprintf("%03d", nodo_id)
          ) :=  as.integer( nodo_id == prediccion[ , arbolito]) ]

       }

       rm( hojas_arbol )
    }
    cat( "\n" )
    flush.console()

    rm( prediccion )
    rm( datamatrix )
    gc(verbose= FALSE)
  }

  gc(verbose= FALSE)

  # borro clase01 , no debe ensuciar el dataset
  dataset[ , clase01 := NULL ]
  cat("Variable clase01 eliminada del dataset\n")
  flush.console()

}

# Parametros de Feature Engineering  a partir de hojas de Random Forest
cat("\n=== Configurando parámetros de Random Forest ===\n")
flush.console()

# Estos CUATRO parametros son los que se deben modificar
PARAM$FE_rf$arbolitos= 20
PARAM$FE_rf$hojas_por_arbol= 16
PARAM$FE_rf$datos_por_hoja= 100
PARAM$FE_rf$mtry_ratio= 0.2

cat("Parámetros RF configurados:\n")
cat("  - Arbolitos:", PARAM$FE_rf$arbolitos, "\n")
cat("  - Hojas por árbol:", PARAM$FE_rf$hojas_por_arbol, "\n")
cat("  - Datos por hoja:", PARAM$FE_rf$datos_por_hoja, "\n")
cat("  - MTRY ratio:", PARAM$FE_rf$mtry_ratio, "\n")
flush.console()

# Estos son quasi fijos
PARAM$FE_rf$train$training <- c(202011, 202012, 202101, 202102, 202103)
cat("Meses de entrenamiento RF:", paste(PARAM$FE_rf$train$training, collapse=", "), "\n")
flush.console()

# Estos TAMBIEN son quasi fijos
PARAM$FE_rf$lgb_param <-list(
    # parametros que se pueden cambiar
    num_iterations = PARAM$FE_rf$arbolitos,
    num_leaves  = PARAM$FE_rf$hojas_por_arbol,
    min_data_in_leaf = PARAM$FE_rf$datos_por_hoja,
    feature_fraction_bynode  = PARAM$FE_rf$mtry_ratio,

    # para que LightGBM emule Random Forest
    boosting = "rf",
    bagging_fraction = ( 1.0 - 1.0/exp(1.0) ),
    bagging_freq = 1.0,
    feature_fraction = 1.0,

    # genericos de LightGBM
    max_bin = 31L,
    objective = "binary",
    first_metric_only = TRUE,
    boost_from_average = TRUE,
    feature_pre_filter = FALSE,
    force_row_wise = TRUE,
    verbosity = -100,
    max_depth = -1L,
    min_gain_to_split = 0.0,
    min_sum_hessian_in_leaf = 0.001,
    lambda_l1 = 0.0,
    lambda_l2 = 0.0,

    pos_bagging_fraction = 1.0,
    neg_bagging_fraction = 1.0,
    is_unbalance = FALSE,
    scale_pos_weight = 1.0,

    drop_rate = 0.1,
    max_drop = 50,
    skip_drop = 0.5,

    extra_trees = FALSE
  )

cat("Parámetros LightGBM configurados\n")
flush.console()

# Feature Engineering agregando variables de Random Forest
#  aqui es donde se hace el trabajo
cat("\nEjecutando AgregaVarRandomForest()...\n")
flush.console()
AgregaVarRandomForest()

cat("\n=== FIN: Feature Engineering con Random Forest completado ===\n")
cat("Total de columnas en dataset:", ncol(dataset), "\n")
print(Sys.time())
flush.console()

ncol(dataset)
colnames(dataset)


# training y future
cat("\n=== INICIO: Preparación datos de entrenamiento (Simulación) ===\n")
print(Sys.time())
flush.console()

PARAM$train$meses <- c(
  201901, 201902, 201903, 201904, 201905, 201906,
  201907, 201908, 201909, 201910, 201911, 201912,
  202001, 202002, 202003, 202004, 202005, 202006,
  202007, 202008, 202009, 202010, 202011, 202012,
  202101, 202102, 202103, 202104
)
PARAM$train$undersampling <- 0.1

PARAM$future <- c(202106)

cat("Meses de entrenamiento:", length(PARAM$train$meses), "\n")
cat("Undersampling:", PARAM$train$undersampling, "\n")
cat("Mes futuro:", PARAM$future, "\n")
flush.console()

# se filtran los meses donde se entrena el modelo
dataset_train <- dataset[foto_mes %in% PARAM$train$meses]
cat("Registros en dataset_train:", nrow(dataset_train), "\n")
flush.console()

# Undersampling, van todos los "BAJA+1" y "BAJA+2" y solo algunos "CONTINIA"
cat("Aplicando undersampling...\n")
flush.console()

set.seed(PARAM$semilla_primigenia, kind = "L'Ecuyer-CMRG")
dataset_train[, azar := runif(nrow(dataset_train))]
dataset_train[, training := 0L]

dataset_train[
  (azar <= PARAM$train$undersampling | clase_ternaria %in% c("BAJA+1", "BAJA+2")),
  training := 1L
]

dataset_train[, azar:= NULL] # elimino la columna azar

cat("Registros seleccionados para training:", dataset_train[training == 1L, .N], "\n")
flush.console()

# paso la clase a binaria que tome valores {0,1}  enteros
#  BAJA+1 y BAJA+2  son  1,   CONTINUA es 0
#  a partir de ahora ya NO puedo cortar  por prob(BAJA+2) > 1/40

dataset_train[,
  clase01 := ifelse(clase_ternaria %in% c("BAJA+2","BAJA+1"), 1L, 0L)
]

cat("Clase binaria creada. Positivos:", dataset_train[training == 1L & clase01 == 1L, .N], "\n")
flush.console()

cat("\n=== Instalando/cargando librerías ===\n")
flush.console()

if (!require("lightgbm", quietly = TRUE)) {
  cat("Instalando lightgbm...\n")
  flush.console()
  install.packages("lightgbm", repos = "https://cloud.r-project.org")
}
cat("lightgbm cargado\n")
print(Sys.time())
flush.console()

# utilizo  zLightGBM  la nueva libreria
if( !require("zlightgbm") ) {
  cat("Instalando zlightgbm...\n")
  flush.console()
  install.packages("https://storage.googleapis.com/open-courses/dmeyf2025-e4a2/zlightgbm_4.6.0.99.tar.gz", repos= NULL, type= "source")
}
require("zlightgbm")
cat("zlightgbm cargado\n")
print(Sys.time())
flush.console()

# canaritos
cat("\n=== Creando canaritos ===\n")
PARAM$qcanaritos <- 100
cat("Cantidad de canaritos:", PARAM$qcanaritos, "\n")
flush.console()

cols0 <- copy(colnames(dataset_train))
filas <- nrow(dataset_train)

for( i in seq(PARAM$qcanaritos) ){
  dataset_train[, paste0("canarito_",i) := runif( filas) ]
}

# las columnas canaritos mandatoriamente van al comienzo del dataset
cols_canaritos <- copy( setdiff( colnames(dataset_train), cols0 ) )
setcolorder( dataset_train, c( cols_canaritos, cols0 ) )

cat("Canaritos creados y reordenados\n")
print(Sys.time())
flush.console()

# los campos que se van a utilizar

campos_buenos <- setdiff(
  colnames(dataset_train),
  c("clase_ternaria", "clase01", "training")
)

cat("\n=== Preparando lgb.Dataset ===\n")
cat("Campos a utilizar:", length(campos_buenos), "\n")
flush.console()

# dejo los datos en el formato que necesita LightGBM

dtrain <- lgb.Dataset(
  data= data.matrix(dataset_train[training == 1L, campos_buenos, with= FALSE]),
  label= dataset_train[training == 1L, clase01],
  free_raw_data= FALSE
)

cat("lgb.Dataset creado - filas:", nrow(dtrain), "columnas:", ncol(dtrain), "\n")
print(Sys.time())
flush.console()

# AGREGAR AQUÍ: Liberar memoria después de crear dtrain
rm(dataset_train)
gc(verbose = TRUE)
cat("Memoria liberada\n")
print(Sys.time())
flush.console()

# definicion de parametros, los viejos y los nuevos
# Voy a entrenar 5 modelos con diferentes semillas

cat("Inicio entrenamiento de modelos\n")
print(Sys.time())

semillas <- c(143287, 146213, 310111, 670627, 880553,100003,  140009,  180023,  220009,  260003, 300007,  340027,  380041,  420047,  460051, 500057,  540061,  580079,  620087,  660097, 700099,  740099,  780119,  820129,  860143, 900149,  940157,  980159,  980179,  999983)

# Preparar estructura para predicciones
predicciones_list <- list()

# Preparar dfuture UNA SOLA VEZ antes del loop
cat("\n=== Preparando datos futuros ===\n")
dfuture <- dataset[foto_mes %in% PARAM$future]
cat("Registros en dfuture:", nrow(dfuture), "\n")
flush.console()

# Agregar canaritos a dfuture
filas <- nrow(dfuture)
for(i in seq(PARAM$qcanaritos)) {
  dfuture[, paste0("canarito_",i) := runif(filas)]
}

# Crear carpeta simul para guardar modelos
simulation_folder <- "simul"
dir.create(simulation_folder, showWarnings = FALSE, recursive = TRUE)

# Inicializar dataset de realidad para medir ganancia
drealidad <- dfuture[, list(numero_de_cliente, foto_mes, clase_ternaria)]

# Función para evaluar ganancia
realidad_evaluar <- function(prealidad, pprediccion) {
  prealidad[pprediccion,
    on= c("numero_de_cliente", "foto_mes"),
    predicted:= i.Predicted
  ]
  
  tbl <- prealidad[, list("qty"=.N), list(predicted, clase_ternaria)]
  
  res <- list()
  res$total <- tbl[predicted==1L, sum(qty*ifelse(clase_ternaria=="BAJA+2", 780000, -20000))]
  
  prealidad[, predicted:=NULL]
  return(res)
}

# Función para calcular ganancia
calcular_ganancia <- function(proba, drealidad) {
  tb_prediccion_temp <- data.table(
    numero_de_cliente = drealidad$numero_de_cliente,
    foto_mes = drealidad$foto_mes,
    proba = proba,
    Predicted = 0L
  )
  
  setorder(tb_prediccion_temp, -proba)
  
  ganancias <- c()
  envios_vec <- seq(500, 20000, by=50)
  
  for(envios in envios_vec) {
    tb_prediccion_temp[, Predicted := 0L]
    tb_prediccion_temp[1:envios, Predicted := 1L]
    ganancia <- realidad_evaluar(copy(drealidad), tb_prediccion_temp)$total
    ganancias <- c(ganancias, ganancia)
  }
  
  idx_max <- which.max(ganancias)
  return(list(
    max_ganancia = ganancias[idx_max],
    envios_optimo = envios_vec[idx_max],
    todas_ganancias = ganancias,
    envios = envios_vec
  ))
}

# LOOP OPTIMIZADO: Entrenar -> Guardar -> Predecir -> Evaluar -> Liberar
for(i in seq_along(semillas)) {
  cat("\n=== Entrenando modelo", i, "con semilla", semillas[i], "===\n")
  print(Sys.time())
  flush.console()
  
  PARAM$lgbm <- list(
    boosting= "gbdt",
    objective= "binary",
    metric= "custom",
    first_metric_only= FALSE,
    boost_from_average= TRUE,
    feature_pre_filter= FALSE,
    force_row_wise= TRUE,
    verbosity= -100,
    seed= semillas[i],
    max_bin= 31L,
    min_data_in_leaf= 20L,
    num_iterations= 9999L,
    num_leaves= 9999L,
    learning_rate= 1.0,
    feature_fraction= 0.50,
    canaritos= PARAM$qcanaritos,
    gradient_bound= 0.1
  )
  
  # Entrenar modelo
  modelo <- lgb.train(
    data= dtrain,
    param= PARAM$lgbm
  )
  
  cat("Fin entrenamiento modelo", i, "\n")
  print(Sys.time())
  flush.console()
  
  # Guardar modelo a disco
  modelo_filename <- sprintf("modelo_%d_semilla_%d.txt", i, semillas[i])
  lgb.save(modelo, file.path(simulation_folder, modelo_filename))
  cat("Modelo guardado:", modelo_filename, "\n")
  flush.console()
  
  # Predecir con este modelo
  cat("Prediciendo con modelo", i, "\n")
  flush.console()
  
  predicciones <- predict(
    modelo,
    data.matrix(dfuture[, campos_buenos, with= FALSE])
  )
  
  # Guardar predicciones en la lista
  predicciones_list[[i]] <- predicciones
  
  # Evaluar ganancia de este modelo individual
  cat("=== Evaluando ganancia modelo", i, "===\n")
  flush.console()
  
  resultado <- calcular_ganancia(predicciones, drealidad)
  
  cat("MÁXIMO modelo", i, ":\n")
  cat("  Envíos óptimos:", resultado$envios_optimo, "\n")
  cat("  Ganancia máxima:", resultado$max_ganancia, "\n")
  flush.console()
  
  # Liberar modelo de memoria
  rm(modelo)
  gc(full = TRUE, verbose = FALSE)
  cat("Memoria liberada después de modelo", i, "\n\n")
  flush.console()
}

cat("\n=== FIN: Todos los modelos entrenados y evaluados ===\n")
print(Sys.time())
flush.console()

# Liberar dtrain
rm(dtrain)
gc(verbose = TRUE)

# Crear tabla de predicciones con todos los modelos
cat("\n=== Guardando predicciones ===\n")
flush.console()

tb_prediccion <- dfuture[, list(numero_de_cliente, foto_mes)]

for(i in seq_along(predicciones_list)) {
  tb_prediccion[, paste0("prob_modelo_", i) := predicciones_list[[i]] ]
}

fwrite(tb_prediccion,
  file= file.path(simulation_folder, "prediccion.txt"),
  sep= "\t"
)
cat("Predicciones guardadas en:", file.path(simulation_folder, "prediccion.txt"), "\n")
flush.console()

# ============================================================================
# GENERACIÓN DE GRÁFICOS
# ============================================================================

cat("\n=== INICIO: Generación de gráficos ===\n")
print(Sys.time())
flush.console()

# Calcular ganancias para cada modelo individual
cat("Calculando ganancias por modelo...\n")
flush.console()

modelos_cols <- paste0("prob_modelo_", 1:length(predicciones_list))
n_modelos <- length(predicciones_list)

# Rango de envíos a probar
envios_vec <- seq(500, 20000, by=50)

# Calcular ganancia para cada modelo
ganancias_por_modelo <- list()

for(i in 1:n_modelos) {
  cat("Calculando ganancias modelo", i, "...\n")
  flush.console()
  
  # Crear tabla temporal con predicciones ordenadas
  tb_temp <- data.table(
    numero_de_cliente = tb_prediccion$numero_de_cliente,
    foto_mes = tb_prediccion$foto_mes,
    proba = predicciones_list[[i]],
    Predicted = 0L
  )
  
  setorder(tb_temp, -proba)
  
  ganancias <- c()
  
  for(envios in envios_vec) {
    tb_temp[, Predicted := 0L]
    tb_temp[1:envios, Predicted := 1L]
    ganancia <- realidad_evaluar(copy(drealidad), tb_temp)$total
    ganancias <- c(ganancias, ganancia)
  }
  
  ganancias_por_modelo[[i]] <- ganancias
  
  idx_max <- which.max(ganancias)
  cat("  Máximo:", ganancias[idx_max], "con", envios_vec[idx_max], "envíos\n")
  flush.console()
}

# Calcular promedio de todos los modelos
cat("\nCalculando promedio de todos los modelos...\n")
flush.console()

tb_prediccion[, prob_promedio := rowMeans(.SD), .SDcols = modelos_cols]

tb_temp <- data.table(
  numero_de_cliente = tb_prediccion$numero_de_cliente,
  foto_mes = tb_prediccion$foto_mes,
  proba = tb_prediccion$prob_promedio,
  Predicted = 0L
)

setorder(tb_temp, -proba)

ganancias_promedio <- c()

for(envios in envios_vec) {
  tb_temp[, Predicted := 0L]
  tb_temp[1:envios, Predicted := 1L]
  ganancia <- realidad_evaluar(copy(drealidad), tb_temp)$total
  ganancias_promedio <- c(ganancias_promedio, ganancia)
}

idx_max_promedio <- which.max(ganancias_promedio)
cat("Promedio máximo:", ganancias_promedio[idx_max_promedio], "con", envios_vec[idx_max_promedio], "envíos\n")
flush.console()

# Crear gráfico con todos los modelos
cat("\n=== Generando gráfico de todos los modelos ===\n")
flush.console()

# Filtrar datos para mostrar solo de 6k a 19k envíos
idx_rango <- which(envios_vec >= 6000 & envios_vec <= 19000)
envios_filtrados <- envios_vec[idx_rango]

png(file.path(simulation_folder, "grafico_ganancia_todos_modelos.png"), 
    width = 1200, height = 800, res = 120)
par(bg = "white", mar = c(5, 4, 4, 4) + 0.1)

# Obtener rango de ganancias para el gráfico
all_ganancias <- unlist(lapply(ganancias_por_modelo, function(x) x[idx_rango]))
all_ganancias <- c(all_ganancias, ganancias_promedio[idx_rango])

# Inicializar el gráfico con límites apropiados
plot(envios_filtrados, ganancias_por_modelo[[1]][idx_rango], 
     type = "n",
     xlim = range(envios_filtrados),
     ylim = range(all_ganancias),
     xlab = "Número de Envíos", 
     ylab = "Ganancia",
     main = "Ganancia vs Número de Envíos - Todos los Modelos")

# Graficar cada modelo individual en gris
for(i in 1:n_modelos) {
  lines(envios_filtrados, 
        ganancias_por_modelo[[i]][idx_rango], 
        col = "grey", 
        lwd = 1)
}

# Graficar el promedio en rojo
lines(envios_filtrados, 
      ganancias_promedio[idx_rango], 
      col = "red", 
      lwd = 3)

# Encontrar el máximo del promedio
max_idx <- which.max(ganancias_promedio[idx_rango])
max_envios <- envios_filtrados[max_idx]
max_ganancia <- ganancias_promedio[idx_rango][max_idx]

# Marcar el punto máximo con una línea vertical
abline(v = max_envios, col = "blue", lwd = 2, lty = 2)

# Imprimir los valores del máximo
cat("Máximo encontrado (Promedio):\n")
cat("Envíos:", max_envios, "\n")
cat("Ganancia:", round(max_ganancia, 0), "\n")

legend("topright", 
       legend = c("Modelos Individuales", "Promedio", "Máximo Promedio"),
       col = c("grey", "red", "blue"), 
       lty = c(1, 1, 2), 
       lwd = c(1, 3, 2),
       bty = "n",
       inset = c(0.02, 0.02))
grid()
dev.off()

cat("Gráfico guardado:", file.path(simulation_folder, "grafico_ganancia_todos_modelos.png"), "\n")
flush.console()

# Calcular moving average para todas las curvas
cat("\n=== Calculando Moving Average k=20 para todos los modelos ===\n")
flush.console()

k <- 20

# Aplicar moving average a cada modelo
ganancias_ma_por_modelo <- list()
for(i in 1:n_modelos) {
  ganancias_ma_por_modelo[[i]] <- rollmean(ganancias_por_modelo[[i]], k = k, fill = NA, align = "center")
}

# Aplicar moving average al promedio
ganancias_ma_promedio <- rollmean(ganancias_promedio, k = k, fill = NA, align = "center")

# Crear gráfico con moving average
cat("\n=== Generando gráfico con Moving Average ===\n")
flush.console()

png(file.path(simulation_folder, "ganancia_moving_average.png"), 
    width = 1200, height = 800, res = 120)

par(bg = "white", mar = c(5, 4, 4, 4) + 0.1)

# Obtener rangos para el gráfico (sin NAs, solo de 6k a 19k)
all_ganancias_ma <- unlist(lapply(ganancias_ma_por_modelo, function(x) x[idx_rango]))
all_ganancias_ma <- c(all_ganancias_ma, ganancias_ma_promedio[idx_rango])
all_ganancias_ma <- all_ganancias_ma[!is.na(all_ganancias_ma)]

plot(envios_filtrados, ganancias_ma_por_modelo[[1]][idx_rango], 
     type = "n",
     xlim = range(envios_filtrados),
     ylim = range(all_ganancias_ma),
     xlab = "Número de Envíos", 
     ylab = "Ganancia (Moving Average k=20)",
     main = "Ganancia vs Número de Envíos - Todos los Modelos (MA)")

# Graficar cada modelo individual en gris
for(i in 1:n_modelos) {
  lines(envios_filtrados, 
        ganancias_ma_por_modelo[[i]][idx_rango], 
        col = "grey", 
        lwd = 1)
}

# Graficar el promedio en rojo
lines(envios_filtrados, 
      ganancias_ma_promedio[idx_rango], 
      col = "red", 
      lwd = 3)

# Encontrar el máximo del promedio (ignorando NAs)
ganancia_promedio_ma_filtrado <- ganancias_ma_promedio[idx_rango]
max_idx_ma <- which.max(ganancia_promedio_ma_filtrado)
max_envios_ma <- envios_filtrados[max_idx_ma]
max_ganancia_ma <- ganancia_promedio_ma_filtrado[max_idx_ma]

# Marcar el punto máximo con una línea vertical
abline(v = max_envios_ma, col = "blue", lwd = 2, lty = 2)

# Imprimir los valores del máximo
cat("Máximo encontrado (Promedio con MA k=20):\n")
cat("Envíos:", max_envios_ma, "\n")
cat("Ganancia:", round(max_ganancia_ma, 0), "\n")

legend("topright", 
       legend = c("Modelos Individuales (MA)", "Promedio (MA)", "Máximo Promedio"),
       col = c("grey", "red", "blue"), 
       lty = c(1, 1, 2), 
       lwd = c(1, 3, 2),
       bty = "n",
       inset = c(0.02, 0.02))
grid()
dev.off()

cat("Gráfico guardado:", file.path(simulation_folder, "ganancia_moving_average.png"), "\n")
flush.console()

cat("\n=== FIN: Gráficos generados exitosamente ===\n")
print(Sys.time())
flush.console()

cat("\n=== SCRIPT COMPLETADO EXITOSAMENTE ===\n")
print(Sys.time())