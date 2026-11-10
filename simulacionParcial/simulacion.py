"""Simulador de las actividades 2-5 y 7-12 del Proyecto 10.

Imprime los resultados, guarda seis gráficas junto al script y ejecuta
comprobaciones básicas del modelo.

Uso: python simulacion.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
## Variables
k = 8.988e9 # Constante de Coulomb (N·m²/C²).
eps_0 = 8.854e-12 # Permitividad del vacío (F/m).

# Se usan para las pruebas de "Verificación Independiente" y la Actividad 8.
# Validan que k y eps_0 sean coherentes entre sí (k * 4 * pi * eps_0 debe ser aprox 1).

PRODUCTO_CONSTANTES = k * 4 * np.pi * eps_0
DESVIACION_CONSTANTES = abs(PRODUCTO_CONSTANTES - 1)
TOL_CONSTANTES = 3.0e-5

# actividades 2-9
R_esfera = 0.05 # Radio fijo de la esfera principal en metros (5 cm).
Q = 1.0e-9 # Carga fija de la esfera en Coulombs (1 nC).

# Cargas de prueba para medir trabajo y energía (Actividad 8)
q0 = 1.0e-9                 # Carga de prueba estándar (1 nC).
q0_experimento_2nC = 2.0e-9 # Carga de prueba modificada (2 nC).
q0_experimento_075nC = 0.75e-9 # Carga de prueba modificada (0.75 nC).

# Arreglos para barridos de datos (Actividades 2, 3, 4, 5 y 7)
# Evalúan instantáneamente múltiples radios sin usar ciclos 'for'.
R_barrido = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
R_destacados = np.array([0.03, 0.05, 0.08])

 # actividades 10, 11 y 12

R_coraza_interna = 0.08 # Radio interior de la coraza externa (8 cm). 'a' es R_esfera.
V_capacitor = 100.0 # Voltaje de la batería conectada al capacitor (100 V).
kappa = 2.0 # Constante dieléctrica del material insertado (kappa o K).

# visualizacion grafica actividad 9

NIVELES_EQUIPOTENCIALES_V = [40, 60, 80, 100, 120, 140, 160]
SEMILADO_VENTANA = 0.25

 # Detecta automáticamente dónde está guardado el script para dejar las imágenes ahí.
RUTA_SALIDA = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# 2. FÍSICA: funciones puras (convención: carga primero, luego radio)
# =====================================================================
def calcular_capacitancia_esfera(R):
    """Capacitancia de una esfera conductora aislada, C = 4π·ε0·R (F)."""
    return 4 * np.pi * eps_0 * R


def calcular_potencial_superficie(Q, R):
    """Potencial de la superficie con V = 0 en el infinito, V = kQ/R (V)."""
    return k * Q / R


def calcular_campo_superficie(Q, R):
    """Campo eléctrico justo afuera de la superficie, E = kQ/R² (V/m)."""
    return k * Q / (R * R)


def calcular_energia_esfera(C, V):
    """Energía almacenada U = ½·C·V² (J). Sirve para cualquier capacitor."""
    return 0.5 * C * V * V


def perfil_potencial(Q, R, r):
    """V(r) de una esfera conductora: constante en r <= R y kQ/r en r > R (V)."""
    r = np.asarray(r, dtype=float)
    # FÍSICA: Dentro de un conductor en equilibrio (r <= R), el campo eléctrico es cero, 
    # por lo que el potencial es constante e igual al de la superficie (kQ/R).
    # Fuera del conductor (r > R), se comporta como una carga puntual (kQ/r).
    # np.maximum evita dividir entre cero cuando r = 0 (ahí se usa la rama interior).
    return np.where(r <= R, k * Q / R, k * Q / np.maximum(r, R))


def perfil_campo(Q, R, r):
    """E(r) de una esfera conductora: 0 en r < R y kQ/r² en r >= R (V/m)."""
    r = np.asarray(r, dtype=float)
    # FÍSICA: Por Ley de Gauss, el campo interno de un conductor sólido es 0.
    # Afuera, decae con el cuadrado de la distancia (kQ/r^2).
    return np.where(r < R, 0.0, k * Q / np.maximum(r, R) ** 2)


def calcular_trabajo_traslado(q0, Q, R, r_i, r_f):
    """Trabajo de la fuerza eléctrica y cambio de energía potencial de q0.

    Mueve q0 de r_i a r_f en el campo de una esfera conductora de radio R y carga Q.
    Usa perfil_potencial, así que también es válida para puntos dentro del conductor.

    Returns:
        (W, delta_U) en julios, con W = -q0·ΔV y ΔU = q0·ΔV.
    """
    delta_V = perfil_potencial(Q, R, r_f) - perfil_potencial(Q, R, r_i)
    # FÍSICA: El trabajo realizado por el campo es W = -q0 * ΔV.
    # El cambio en la energía potencial del sistema es ΔU = q0 * ΔV.
    # Si W es positivo, el campo hace el trabajo. Si W es negativo, lo hace un agente externo.
    return float(-q0 * delta_V), float(q0 * delta_V)


def capacitancia_capacitor_esferico(a, b, kappa=1.0):
    """C = κ·4π·ε0·a·b/(b - a) de dos cascarones concéntricos, a < b (F)."""
    if np.any(np.asarray(b) <= a):
        raise ValueError("Se requiere a < b (radio interior menor que el exterior).")
    return kappa * 4 * np.pi * eps_0 * (a * b / (b - a))


def calcular_estado_capacitor(C, V):
    """Carga Q = C·V (C) y energía U = ½·C·V² (J) de un capacitor a tensión V."""
    return C * V, calcular_energia_esfera(C, V)


# =====================================================================
# 3. UTILIDADES
# =====================================================================
def pendiente_loglog(x, y):
    """Exponente n de la ley de potencias y ∝ x^n (pendiente en escala log-log)."""
    # FÍSICA/MATE: Si una variable depende de otra según y = A*x^n, 
    # al aplicar logaritmos queda log(y) = log(A) + n*log(x).
    # Esta función calcula la pendiente 'n', demostrando numéricamente si 
    # algo es inversamente proporcional (n = -1) o lineal (n = 1).
    return np.polyfit(np.log(x), np.log(y), 1)[0]


def diferencia_relativa(medido, referencia):
    """|medido - referencia| / |referencia|."""
    return abs(medido - referencia) / abs(referencia)


def _marcar_destacados(eje, radios_cm, valores):
    """Marca en rojo los radios 3, 5 y 8 cm sobre una curva del barrido."""
    for radio_cm in R_destacados * 100:
        i = np.argmin(np.abs(radios_cm - radio_cm))
        eje.scatter(radios_cm[i], valores[i], color="red", zorder=3)
        eje.annotate(f"{radio_cm:.0f} cm", (radios_cm[i], valores[i]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8)


def _guardar(fig, nombre):
    """Guarda la figura como PNG en RUTA_SALIDA y la cierra."""
    fig.savefig(os.path.join(RUTA_SALIDA, nombre), dpi=300)
    plt.close(fig)
    print(f"Gráfica guardada: {nombre}")


# =====================================================================
# 4. ACTIVIDADES
# =====================================================================
def actividades_2_3_4():
    """Barrido de R con Q fija: C, V, E y U (más pendientes log-log)."""
    C = calcular_capacitancia_esfera(R_barrido)
    V = calcular_potencial_superficie(Q, R_barrido)
    E = calcular_campo_superficie(Q, R_barrido)
    U = calcular_energia_esfera(C, V)

    print(f"\nActividades 2-4: barrido de R con Q fija (Q = {Q * 1e9:.1f} nC)")
    print(f"{'R (cm)':>8} {'C (pF)':>10} {'V (V)':>10} {'E (kV/m)':>10} {'U (nJ)':>10}")
    for i, R in enumerate(R_barrido):
        print(f"{R * 100:8.1f} {C[i] * 1e12:10.3f} {V[i]:10.2f} "
              f"{E[i] / 1e3:10.2f} {U[i] * 1e9:10.3f}")
    pendientes = {n: pendiente_loglog(R_barrido, y)
                  for n, y in (("C", C), ("V", V), ("E", E), ("U", U))}
    print("Exponentes de R (pendiente log-log): "
          + ", ".join(f"{n} ∝ R^{p:+.3f}" for n, p in pendientes.items()))

    fig, ejes = plt.subplots(2, 2, figsize=(11, 8))
    datos = [(C * 1e12, "C (pF)", "Capacitancia C(R)"),
             (V, "V (V)", "Potencial de la superficie V(R)"),
             (E / 1e3, "E (kV/m)", "Campo justo afuera E(R)"),
             (U * 1e9, "U (nJ)", "Energía almacenada U(R)")]
    for eje, (valores, etiqueta, titulo) in zip(ejes.flat, datos):
        eje.plot(R_barrido * 100, valores, marker="o")
        _marcar_destacados(eje, R_barrido * 100, valores)
        eje.set_title(titulo)
        eje.set_xlabel("Radio R (cm)")
        eje.set_ylabel(etiqueta)
        eje.grid(True, alpha=0.3)
    fig.suptitle("Barrido del radio con carga fija Q = 1 nC")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _guardar(fig, "barrido_R_Q_fija.png")


def actividad_5():
    """V(r) y E(r) para los radios 3, 5 y 8 cm en los mismos ejes (Q fija)."""
    r = np.linspace(0.005, 0.20, 2000)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for R in R_destacados:
        etiqueta = f"R = {R * 100:.0f} cm"
        linea, = ejes[0].plot(r * 100, perfil_potencial(Q, R, r), label=etiqueta)
        ejes[1].plot(r * 100, perfil_campo(Q, R, r) / 1e3,
                     color=linea.get_color(), label=etiqueta)
        for eje in ejes:
            eje.axvline(R * 100, color=linea.get_color(), linestyle=":", alpha=0.5)
    ejes[0].set_title("Potencial V(r): constante adentro, kQ/r afuera")
    ejes[0].set_ylabel("V (V)")
    ejes[1].set_title("Campo E(r): cero adentro, kQ/r² afuera")
    ejes[1].set_ylabel("E (kV/m)")
    for eje in ejes:
        eje.set_xlabel("Distancia radial r (cm)")
        eje.grid(True, alpha=0.3)
        eje.legend()
    fig.suptitle("Perfiles comparativos con Q = 1 nC (las líneas punteadas marcan r = R)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _guardar(fig, "perfiles_comparativos_tres_radios.png")


def actividad_7():
    """Compara carga fija (Q = 1 nC) con potencial fijo (V de la esfera de 5 cm)."""
    V_fijo = calcular_potencial_superficie(Q, R_esfera)
    C = calcular_capacitancia_esfera(R_barrido)

    # Caso A: Q fija
    V_a = calcular_potencial_superficie(Q, R_barrido)
    E_a = calcular_campo_superficie(Q, R_barrido)
    U_a = calcular_energia_esfera(C, V_a)
    # Caso B: V fijo
    Q_b = C * V_fijo
    E_b = calcular_campo_superficie(Q_b, R_barrido)
    U_b = calcular_energia_esfera(C, V_fijo)

    # - Con Q fija: Al aumentar el radio, la capacitancia (C) crece, por lo que el potencial (V=Q/C) cae.
    # - Con V fijo: Para mantener el mismo voltaje en una esfera más grande (mayor C), 
    #   la batería debe suministrar más carga (Q=C*V). 
    # Las curvas se cruzan exactamente en R = 5 cm porque usamos el voltaje de esa esfera como referencia.
    print(f"\nActividad 7: potencial fijo V = {V_fijo:.2f} V "
          f"(el de la esfera de {R_esfera * 100:.0f} cm con Q = 1 nC)")
    print(f"{'R (cm)':>8} {'C (pF)':>10} {'Q (nC)':>10} {'V (V)':>10} "
          f"{'E (kV/m)':>10} {'U (nJ)':>10}")
    for i, R in enumerate(R_barrido):
        print(f"{R * 100:8.1f} {C[i] * 1e12:10.3f} {Q_b[i] * 1e9:10.3f} {V_fijo:10.2f} "
              f"{E_b[i] / 1e3:10.2f} {U_b[i] * 1e9:10.3f}")
    print("Exponentes de R con V fijo: "
          f"Q ∝ R^{pendiente_loglog(R_barrido, Q_b):+.3f}, "
          f"E ∝ R^{pendiente_loglog(R_barrido, E_b):+.3f}, "
          f"U ∝ R^{pendiente_loglog(R_barrido, U_b):+.3f}")
    print("Las dos condiciones coinciden en R = 5 cm porque V fijo se tomó de esa esfera.")

    fig, ejes = plt.subplots(2, 2, figsize=(11, 8))
    paneles = [
        (np.full(R_barrido.size, Q * 1e9), Q_b * 1e9, "Carga Q (nC)", "Carga"),
        (V_a, np.full(R_barrido.size, V_fijo), "V (V)", "Potencial de la superficie"),
        (E_a / 1e3, E_b / 1e3, "E (kV/m)", "Campo justo afuera"),
        (U_a * 1e9, U_b * 1e9, "U (nJ)", "Energía almacenada"),
    ]
    for eje, (y_a, y_b, etiqueta, titulo) in zip(ejes.flat, paneles):
        eje.plot(R_barrido * 100, y_a, "o-", label="Q fija")
        eje.plot(R_barrido * 100, y_b, "s-", label="V fijo")
        eje.axvline(R_esfera * 100, color="gray", linestyle=":", alpha=0.6)
        eje.set_title(titulo)
        eje.set_xlabel("Radio R (cm)")
        eje.set_ylabel(etiqueta)
        eje.grid(True, alpha=0.3)
        eje.legend()
    fig.suptitle("Actividad 7: carga fija frente a potencial fijo "
                 "(línea punteada: R = 5 cm, donde se cruzan)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _guardar(fig, "comparacion_Qfija_Vfijo.png")


def _interpretar_signo(W):
    """Texto que interpreta el signo de W y de ΔU = -W."""
    if W > 0:
        return ("W > 0 y ΔU = -W < 0: el campo realiza trabajo y la energía potencial "
                "de la carga de prueba disminuye.")
    if W < 0:
        return ("W < 0 y ΔU = -W > 0: un agente externo debe realizar trabajo; la energía "
                "potencial de la carga de prueba aumenta.")
    return "W = 0 y ΔU = 0: ambos puntos están al mismo potencial."


def experimento_actividad_8(nombre, carga_prueba, radio_destino, Q_esfera=Q, R=R_esfera):
    """Compara el trabajo W de la fuerza eléctrica sobre q0 con la energía U de la esfera."""
    W, delta_U = calcular_trabajo_traslado(carga_prueba, Q_esfera, R, R, radio_destino)
    U = calcular_energia_esfera(calcular_capacitancia_esfera(R),
                                calcular_potencial_superficie(Q_esfera, R))
    razon_medida = W / U
    # Álgebra: W = q0·kQ(1/R - 1/Rd) y U = ½·(k·4π·ε0)·kQ²/R
    razon_algebra = ((carga_prueba / Q_esfera) * 2 * (1 - R / radio_destino)
                     / PRODUCTO_CONSTANTES)
    diferencia = diferencia_relativa(razon_medida, razon_algebra)

    print(f"\n{nombre}  (q0 = {carga_prueba * 1e9:.2f} nC, de R = {R * 100:.1f} cm "
          f"a {radio_destino * 100:.1f} cm)")
    print(f"  W (fuerza eléctrica) = {W * 1e9:9.3f} nJ    "
          f"ΔU de q0 = {delta_U * 1e9:9.3f} nJ")
    print(f"  U almacenada en la esfera = {U * 1e9:9.3f} nJ")
    print(f"  W/U según el álgebra = {razon_algebra:.6f}    "
          f"W/U medida = {razon_medida:.6f}    diferencia relativa = {diferencia:.1e}")
    print(f"  {_interpretar_signo(W)}")


def actividad_8():
    """Trabajo de una carga de prueba entre R y 2R, y qué la distingue de la U de la esfera."""
    print("\nActividad 8: W = q0·[V(R) - V(destino)] (trabajo de la fuerza eléctrica)  "
          "frente a  U = ½·C·V² (energía de la esfera)")
    print("W/U = (q0/Q)·2·(1 - R/destino)/(k·4π·ε0), con k·4π·ε0 = "
          f"{PRODUCTO_CONSTANTES:.9f}")
    experimento_actividad_8("Datos oficiales: q0 = 1 nC, destino 2R", q0, 2 * R_esfera)
    experimento_actividad_8("Experimento: q0 = 2 nC, destino 2R",
                            q0_experimento_2nC, 2 * R_esfera)
    experimento_actividad_8("Experimento: q0 = 1 nC, destino 3R", q0, 3 * R_esfera)
    experimento_actividad_8("Experimento: q0 = 0.75 nC, destino 3R",
                            q0_experimento_075nC, 3 * R_esfera)
    print("\nCon destino 2R, V(2R) = V(R)/2, así que W = ½·q0·V(R) mientras que "
          "U = ½·Q·V(R): coinciden (W/U ≈ 1) cuando q0 = Q. Pero W/U ≈ 1 también "
          "ocurre con otras parejas (q0, destino), como la última. W es el trabajo "
          "del campo sobre q0; U es la energía asociada a la carga Q de la esfera.")


def actividad_9():
    """Superficies equipotenciales de la esfera de 5 cm (vista 2D) con niveles elegidos."""
    x = np.linspace(-SEMILADO_VENTANA, SEMILADO_VENTANA, 500)
    X, Y = np.meshgrid(x, x)
    r_plano = np.hypot(X, Y)
    V_plano = perfil_potencial(Q, R_esfera, r_plano)

    niveles = np.asarray(NIVELES_EQUIPOTENCIALES_V, dtype=float)
    if (niveles.size == 0 or np.any(np.diff(niveles) <= 0)
            or niveles[0] <= V_plano.min() or niveles[-1] >= V_plano.max()):
        raise ValueError("NIVELES_EQUIPOTENCIALES_V debe ser ascendente y quedar entre "
                         f"{V_plano.min():.1f} V y {V_plano.max():.1f} V.")
    radios_nivel = k * Q / niveles * 100
    print("\nActividad 9: equipotenciales en "
          + ", ".join(f"{v:.0f} V (r = {r:.1f} cm)" for v, r in zip(niveles, radios_nivel)))

    fig, eje = plt.subplots(figsize=(7, 7))
    contornos = eje.contour(X * 100, Y * 100, V_plano, levels=niveles, cmap="plasma")
    eje.clabel(contornos, inline=True, fontsize=9, fmt="%.0f V")
    for i, angulo in enumerate(np.deg2rad(np.arange(0, 360, 45))):
        radios = np.array([R_esfera, SEMILADO_VENTANA - 0.01]) * 100
        eje.plot(radios * np.cos(angulo), radios * np.sin(angulo), "k--",
                 linewidth=0.7, alpha=0.5, label="Líneas de campo" if i == 0 else None)
    eje.add_patch(plt.Circle((0, 0), R_esfera * 100, color="silver", ec="black",
                             zorder=10, label="Esfera conductora"))
    eje.set_title("Equipotenciales (círculos) y líneas de campo (radiales): "
                  "siempre perpendiculares")
    eje.set_xlabel("x (cm)")
    eje.set_ylabel("y (cm)")
    eje.set_aspect("equal")
    eje.grid(True, linestyle=":", alpha=0.6)
    eje.legend(loc="upper right")
    fig.tight_layout()
    _guardar(fig, "superficies_equipotenciales.png")


def actividades_10_11_12():
    """Capacitor esférico (a = 5 cm, b = 8 cm) y efecto del dieléctrico en dos condiciones."""
    a, b = R_esfera, R_coraza_interna
    C0 = capacitancia_capacitor_esferico(a, b)
    Q0, U0 = calcular_estado_capacitor(C0, V_capacitor)
    print(f"\nActividades 10-11: capacitor esférico en vacío, a = {a * 100:.1f} cm "
          f"(la esfera), b = {b * 100:.1f} cm (radio interior de la coraza), "
          f"V = {V_capacitor:.0f} V")
    print(f"  C = {C0 * 1e12:.3f} pF    Q = {Q0 * 1e9:.3f} nC    U = {U0 * 1e9:.3f} nJ")

    Ck = capacitancia_capacitor_esferico(a, b, kappa)
    Q_con, U_con = calcular_estado_capacitor(Ck, V_capacitor)  # batería conectada: V fijo
    V_con = V_capacitor
    V_des = Q0 / Ck                                            # desconectada: Q fija
    U_des = calcular_energia_esfera(Ck, V_des)
    # Al insertar el dieléctrico, sus moléculas se polarizan y crean un campo eléctrico interno 
    # opuesto al original. Esto "debilita" el campo total.
    # - Batería conectada (V fijo): La fuente bombea más carga para compensar esa caída de campo. (C y Q suben).
    # - Batería desconectada (Q fija): La carga no tiene a dónde huir. Como el campo se debilita, 
    #   el voltaje entre las placas cae irremediablemente. (C sube, V y U caen).
    print(f"\nActividad 12: dieléctrico κ = {kappa:g}")
    print("Predicción (tabla de clase): conectada -> C, Q, U × κ y V igual; "
          "desconectada -> C × κ, Q igual, V y U ÷ κ.")
    print(f"{'Magnitud':<8} {'Vacío':>12} {'κ, conectada':>14} {'κ, desconect.':>15}")
    filas = [("C (pF)", C0 * 1e12, Ck * 1e12, Ck * 1e12),
             ("Q (nC)", Q0 * 1e9, Q_con * 1e9, Q0 * 1e9),
             ("V (V)", V_capacitor, V_con, V_des),
             ("U (nJ)", U0 * 1e9, U_con * 1e9, U_des * 1e9)]
    for nombre, vacio, conectada, desconectada in filas:
        print(f"{nombre:<8} {vacio:12.3f} {conectada:14.3f} {desconectada:15.3f}")
    razones_con = [f[2] / f[1] for f in filas]
    razones_des = [f[3] / f[1] for f in filas]
    print("Razones respecto al vacío (C, Q, V, U): conectada "
          + ", ".join(f"{x:.3f}" for x in razones_con) + " | desconectada "
          + ", ".join(f"{x:.3f}" for x in razones_des))

    posiciones = np.arange(4)
    fig, eje = plt.subplots(figsize=(8, 4.5))
    for desplazamiento, razones, etiqueta in ((-0.2, razones_con, "Batería conectada (V fijo)"),
                                              (0.2, razones_des, "Batería desconectada (Q fija)")):
        barras = eje.bar(posiciones + desplazamiento, razones, width=0.4, label=etiqueta)
        for barra, valor in zip(barras, razones):
            eje.annotate(f"{valor:.2f}", (barra.get_x() + barra.get_width() / 2, valor),
                         xytext=(0, 3), textcoords="offset points", ha="center")
    eje.axhline(1, color="gray", linestyle=":")
    eje.set_xticks(posiciones, ["C", "Q", "V", "U"])
    eje.set_ylabel("Valor / valor en vacío")
    eje.set_ylim(0, 2.6)  # espacio sobre las barras para que la leyenda no tape las etiquetas
    eje.set_title(f"Actividad 12: efecto de insertar un dieléctrico κ = {kappa:g}")
    eje.legend(loc="upper center", ncol=2)
    eje.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _guardar(fig, "actividad_12_dielectrico.png")


def grafica_convergencia_capacitor():
    """Muestra que el capacitor esférico tiende a la esfera aislada cuando b >> a."""
    a = R_esfera
    # La fórmula del capacitor esférico es C = 4π·ε0·(a·b)/(b - a).
    # Si alejamos la coraza exterior hacia el infinito (b → ∞), el término b/(b-a) tiende a 1.
    # Por lo tanto, recuperamos la fórmula de la esfera aislada: C = 4π·ε0·a.
    # Una esfera aislada es simplemente un capacitor donde la segunda placa está en el infinito.
    C_aislada = calcular_capacitancia_esfera(a)
    cociente = np.logspace(np.log10(1.2), 4, 200)  # b/a
    desviacion = (capacitancia_capacitor_esferico(a, a * cociente) - C_aislada) / C_aislada
    cociente_proyecto = R_coraza_interna / a
    desviacion_proyecto = 1 / (cociente_proyecto - 1)

    print("\nConvergencia: (C_capacitor - C_esfera)/C_esfera = a/(b - a); "
          f"baja de 1e-3 cuando b/a > {1 + 1e3:.0f}")
    fig, eje = plt.subplots(figsize=(6.5, 4.5))
    eje.loglog(cociente, desviacion, label="Capacitor esférico (calculado)")
    eje.loglog(cociente, 1 / (cociente - 1), "--", label="Predicción a/(b - a)")
    eje.scatter([cociente_proyecto], [desviacion_proyecto], color="red", zorder=3,
                label="Caso del proyecto (b/a = 1.6)")
    eje.set_xlabel("b / a")
    eje.set_ylabel("Desviación relativa respecto a la esfera aislada")
    eje.set_title("El capacitor tiende a la esfera aislada cuando b → ∞")
    eje.grid(True, which="both", alpha=0.3)
    eje.legend()
    fig.tight_layout()
    _guardar(fig, "convergencia_capacitor_esferico.png")


# Comprobaciones físicas
def ejecutar_pruebas():
    """Comprueba algunas relaciones físicas del modelo."""
    resultados = {}
    capacitancia = calcular_capacitancia_esfera(R_esfera)
    potencial = calcular_potencial_superficie(Q, R_esfera)
    energia = calcular_energia_esfera(capacitancia, potencial)
    campo_gauss = Q / (4 * np.pi * R_esfera**2 * eps_0)
    resultados["esfera: potencial, campo y energía"] = all((
        diferencia_relativa(Q / capacitancia, potencial) <= TOL_CONSTANTES,
        diferencia_relativa(calcular_campo_superficie(Q, R_esfera), campo_gauss)
        <= TOL_CONSTANTES,
        diferencia_relativa(energia, 0.5 * Q * potencial) <= TOL_CONSTANTES,
    ))
    # PRUEBA 2: Gradiente de Potencial (Verificación Independiente Principal)
    radio = R_esfera
    r = np.linspace(1.2 * radio, 3 * radio, 1001)
    # FÍSICA: Se calcula la derivada numérica del potencial respecto al radio (dV/dr).
    # Como E = -dV/dr, este resultado numérico debe ser casi idéntico al perfil de campo teórico.
    campo_derivado = -np.gradient(perfil_potencial(Q, radio, r), r)
    campo_modelo = perfil_campo(Q, radio, r)
    error_campo = np.max(
        np.abs(campo_derivado[1:-1] - campo_modelo[1:-1]) / campo_modelo[1:-1]
    )
    resultados["perfiles: E = -dV/dr"] = error_campo < 2e-5

    trabajo, cambio_energia = calcular_trabajo_traslado(
        q0, Q, radio, radio, 2 * radio
    )
    trabajo_desde_dentro, _ = calcular_trabajo_traslado(
        q0, Q, radio, radio / 2, 2 * radio
    )
    resultados["trabajo: signo y recorrido interior"] = (
        trabajo > 0
        and np.isclose(trabajo, trabajo_desde_dentro)
        and np.isclose(trabajo, -cambio_energia)
    )

    a, b = R_esfera, R_coraza_interna
    C_vacio = capacitancia_capacitor_esferico(a, b)
    C_dielec = capacitancia_capacitor_esferico(a, b, kappa)
    Q_vacio, U_vacio = calcular_estado_capacitor(C_vacio, V_capacitor)
    Q_con, U_con = calcular_estado_capacitor(C_dielec, V_capacitor)
    V_des = Q_vacio / C_dielec
    U_des = calcular_energia_esfera(C_dielec, V_des)

    # PRUEBA 4: Comportamiento del Dieléctrico
    # FÍSICA: Batería conectada (V fijo) -> C, Q y U aumentan por factor kappa.
    # Batería desconectada (Q fija) -> C aumenta, pero V y U disminuyen por factor 1/kappa.
    resultados["capacitor: dieléctrico conectado/desconectado"] = all((
        np.isclose(C_dielec / C_vacio, kappa),
        np.isclose(Q_con / Q_vacio, kappa),
        np.isclose(U_con / U_vacio, kappa),
        np.isclose(V_des / V_capacitor, 1 / kappa),
        np.isclose(U_des / U_vacio, 1 / kappa),
    ))

    C_aislada = calcular_capacitancia_esfera(a)
    b_sobre_a = np.array([10.0, 100.0, 1000.0])
    diferencia = (
        capacitancia_capacitor_esferico(a, a * b_sobre_a) - C_aislada
    ) / C_aislada
    resultados["capacitor: límite de esfera aislada"] = np.allclose(
        diferencia, 1 / (b_sobre_a - 1)
    )

    for nombre, aprobada in resultados.items():
        print(f"[{'APROBADA' if aprobada else 'FALLIDA'}] {nombre}")
    return resultados


def main():
    """Ejecuta las actividades y comprobaciones del simulador."""
    print("=" * 70)
    print(" PROYECTO 10: esfera conductora aislada y capacitor esférico")
    print("=" * 70)
    print(f"k = {k:.4e} N·m²/C²   ε0 = {eps_0:.4e} F/m   "
          f"k·4π·ε0 = {PRODUCTO_CONSTANTES:.9f} (desviación {DESVIACION_CONSTANTES:.2e})")
    actividades_2_3_4()
    actividad_5()
    actividad_7()
    actividad_8()
    actividad_9()
    actividades_10_11_12()
    grafica_convergencia_capacitor()
    print("\n=== Verificación independiente ===")
    resultados = ejecutar_pruebas()
    aprobadas = sum(resultados.values())
    print(f"\nPruebas aprobadas: {aprobadas}/{len(resultados)}")


if __name__ == "__main__":
    main()