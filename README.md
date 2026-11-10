# Proyecto 10 – Esfera conductora aislada (Física II, UTP)

Simulador en Python de las 12 actividades del Proyecto 10: barridos de radio, perfiles
V(r) y E(r), trabajo de una carga de prueba, equipotenciales, capacitor esférico y
dieléctrico (batería conectada y desconectada), con pruebas independientes y
verificación por mutación.

## Cómo ejecutarlo (GitHub Codespaces)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python simulador_esfera_conductora.py
```

El script imprime las tablas y las pruebas, y guarda los PNG en la misma carpeta.

## Archivos que genera

- `barrido_R_Q_fija.png` – C, V, E y U contra R con Q fija (actividades 2-4)
- `perfiles_comparativos_tres_radios.png` – V(r) y E(r) para 3, 5 y 8 cm (actividad 5)
- `comparacion_Qfija_Vfijo.png` – carga fija frente a potencial fijo (actividad 7)
- `superficies_equipotenciales.png` – equipotenciales y líneas de campo (actividad 9)
- `actividad_12_dielectrico.png` – dieléctrico con batería conectada y desconectada
- `convergencia_capacitor_esferico.png` – el capacitor tiende a la esfera aislada si b → ∞

## Cambiar los datos

Todas las entradas están en la sección 1 del script (radio, cargas, κ, niveles de
las equipotenciales). Las funciones de física no leen variables globales de datos,
salvo las constantes k y ε0.