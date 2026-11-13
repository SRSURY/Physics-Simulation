# Proyecto 10 – Simulación de campo eléctrico y capacitancia

Este proyecto simula una esfera conductora y un capacitor esférico para estudiar:
- potencial eléctrico,
- campo eléctrico,
- energía almacenada,
- trabajo sobre una carga de prueba,
- equipotenciales,
- efecto de un dieléctrico,
- comparación entre carga fija y potencial fijo.

Está pensado para las actividades 2–12 del Proyecto 10 de Física II.

## ¿Qué hace este código?

El archivo `simulacionParcial/simulacion.py` hace lo siguiente:

1. Define constantes físicas importantes:
   - `k` = constante de Coulomb
   - `eps_0` = permitividad del vacío

2. Crea funciones para calcular:
   - capacitancia de una esfera,
   - potencial y campo eléctrico,
   - energía almacenada,
   - trabajo realizado por el campo,
   - comportamiento de un capacitor esférico con dieléctrico.

3. Genera varias gráficas con `matplotlib` para visualizar:
   - cómo cambian C, V, E y U al variar el radio,
   - perfiles de potencial y campo en distintos radios,
   - equipotenciales y líneas de campo,
   - efecto de introducir un dieléctrico,
   - convergencia del capacitor esférico a una esfera aislada.

4. Ejecuta pruebas de verificación para comprobar que las ecuaciones físicas se cumplen.

## Explicación sencilla del código

La parte más importante es esta:

- Las funciones como `calcular_capacitancia_esfera`, `calcular_potencial_superficie` y `calcular_campo_superficie` no hacen magia: solo aplican fórmulas físicas.
- Después, las funciones de cada actividad usan esas fórmulas para producir tablas y gráficos.
- Por ejemplo:
  - `perfil_potencial(Q, R, r)` calcula el potencial en cada distancia `r`.
  - `perfil_campo(Q, R, r)` calcula el campo eléctrico en función de `r`.
  - `calcular_trabajo_traslado(...)` compara el trabajo del campo con la energía potencial de la carga de prueba.
- Todo está organizado por actividades, para que cada bloque del script responda a una parte del proyecto.

En resumen: el código toma datos físicos, aplica las ecuaciones de electrostática y luego presenta resultados en consola y en imágenes.

## Cómo ejecutarlo


python simulacionParcial/simulacion.py
```

Al ejecutarse, el programa:
- imprime tablas con resultados,
- muestra comprobaciones físicas,
- guarda varios archivos `.png` en la misma carpeta del script.

## Archivos que genera

- `barrido_R_Q_fija.png` – variación de C, V, E y U con el radio
- `perfiles_comparativos_tres_radios.png` – perfiles V(r) y E(r) para radios 3, 5 y 8 cm
- `comparacion_Qfija_Vfijo.png` – comparación entre carga fija y potencial fijo
- `superficies_equipotenciales.png` – equipotenciales y líneas de campo
- `actividad_12_dielectrico.png` – efecto del dieléctrico con batería conectada/desconectada
- `convergencia_capacitor_esferico.png` – cómo el capacitor tiende a una esfera aislada

## Cambiar los parámetros

Las variables principales están al inicio del archivo `simulacion.py`, por ejemplo:
- radio de la esfera,
- carga de la esfera,
- tensión del capacitor,
- constante dieléctrica `kappa`,
- niveles de equipotenciales.

Si quieres modificar la simulación, solo tienes que ajustar esos valores al principio.

## Estructura general del archivo

- Sección de constantes físicas
- Funciones básicas de electrostática
- Utilidades para graficar
- Actividades del proyecto
- Pruebas de validación
- Función `main()` que ejecuta todo

Esto hace que el código sea fácil de seguir: primero se definen las fórmulas, luego se usan para generar resultados y gráficos.
