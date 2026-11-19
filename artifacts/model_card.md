# Tarjeta del modelo — Cobertura forestal (Forest CoverType)

**Versión:** 1.0.0 · **Modelo desplegado:** SVC-RBF calibrado (sigmoid, `CalibratedClassifierCV`, cv=5) · **Alternativo:** HGB calibrado (sigmoid)

## Uso previsto
Apoyo a la cartografía de cobertura forestal en zonas con variables cartográficas como las del Roosevelt National Forest (Colorado, EE. UU.).
Las predicciones son **insumos** para decisiones de conservación y ordenamiento, siempre acompañadas del conjunto de predicción conforme.
**No** debe usarse fuera de la región de origen sin revalidar, ni como única fuente para decisiones irreversibles.

## Datos
UCI Covertype: 581 012 parcelas de 30 × 30 m, 54 variables (10 continuas, 4 áreas silvestres, 40 tipos de suelo), 7 clases desbalanceadas
(razón 103:1). SHA-256 `614360d0257557dd…`. Partición por **bloques pseudo-espaciales** (KMeans k=20); prueba = bloques no vistos.

## Métricas (TEST, bloques no vistos)
| Métrica | Valor |
|---|---|
| macro-F1 | 0.403 |
| balanced accuracy | 0.456 |
| MCC | 0.416 |
| AUC macro (OvR) | 0.870 |
| macro-F1 en IID (misma distribución) | 0.697 |
| ECE (15 intervalos, calibrado) | 0.135 |
| Cobertura conforme α=0.1 (IID / TEST) | 0.900 / 0.816 |

Recall por clase en TEST: Spruce/Fir: 0.59, Lodgepole Pine: 0.73, Ponderosa Pine: 0.27, Cottonwood/Willow: 0.00, Aspen: 0.00, Douglas-fir: 0.44, Krummholz: 0.70

## Limitaciones
- Los bloques son pseudo-espaciales (no hay coordenadas): aproximan, no garantizan, independencia espacial.
- El desempeño cae al pasar de IID a bloques nuevos; la cobertura conforme solo está garantizada bajo intercambiabilidad.
- Clases raras (Cottonwood/Willow, Aspen) tienen menor recall y cobertura condicional; usar el conjunto conforme o Mondrian.
- La SVM exacta se entrenó con 20,000 filas por su costo ≈ cuadrático; HGB con 150,000 (límite de memoria).

## Consideraciones éticas
Errores en coberturas raras (riparias como Cottonwood/Willow) pueden ocultar hábitats sensibles: no deben tomarse decisiones de
aprovechamiento con base solo en el modelo. El monitoreo de deriva (PSI/KS/JS) debe estar activo y la política de reentrenamiento documentada.
