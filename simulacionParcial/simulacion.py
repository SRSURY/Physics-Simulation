"""Proyecto 10 - Esfera conductora aislada y capacitor esférico (Física II, UTP).

Simulador del segundo examen parcial (ABP). Cubre las actividades 1-12 del proyecto:
barridos de radio, perfiles V(r) y E(r), trabajo de una carga de prueba, equipotenciales,
capacitor esférico, dieléctrico (batería conectada y desconectada) y un conjunto de
pruebas independientes con verificación por mutación.

Uso:
    python simulador_esfera_conductora.py

Imprime las tablas y guarda las figuras (PNG) en la misma carpeta del script.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # solo guarda PNG: funciona en Codespaces sin ventanas
import matplotlib.pyplot as plt
from scipy.integrate import quad

# =====================================================================
# 1. CONSTANTES Y DATOS DE DISEÑO (todo en unidades SI)
# =====================================================================
k = 8.988e9        # N·m²/C²  constante de Coulomb (≈ 1/(4π·ε0))
eps_0 = 8.854e-12  # F/m      permitividad eléctrica del vacío

# k y ε0 están redondeados por separado, así que k·4π·ε0 no vale exactamente 1.
# Esa desviación (≈ 2.9e-5) aparece en toda prueba que mezcle ambas constantes.
PRODUCTO_CONSTANTES = k * 4 * np.pi * eps_0
DESVIACION_CONSTANTES = abs(PRODUCTO_CONSTANTES - 1)
TOL_CONSTANTES = 3.0e-5  # tolerancia relativa: justo por encima de la desviación

R_esfera = 0.05                 # m   radio de la esfera (5 cm)
Q = 1.0e-9                      # C   carga de la esfera (1 nC)
q0 = 1.0e-9                     # C   carga de prueba oficial (+1 nC)
q0_experimento_2nC = 2.0e-9     # C   solo para el experimento de la actividad 8
q0_experimento_075nC = 0.75e-9  # C   solo para el experimento de la actividad 8

R_barrido = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])  # m
R_destacados = np.array([0.03, 0.05, 0.08])  # m   radios que pide el proyecto

R_coraza_interna = 0.08  # m   radio INTERIOR de la coraza (actividad 10); es b
V_capacitor = 100.0      # V   diferencia de potencial del capacitor (actividad 11)
kappa = 2.0              # --  constante dieléctrica (actividad 12)

# Niveles de las equipotenciales (V). Se eligen equiespaciados cada 20 V:
# el espaciado entre curvas es inversamente proporcional a |E|.
NIVELES_EQUIPOTENCIALES_V = [40, 60, 80, 100, 120, 140, 160]
SEMILADO_VENTANA = 0.25  # m   la ventana del mapa es [-25, 25] cm en x e y

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
    # np.maximum evita dividir entre cero cuando r = 0 (ahí se usa la rama interior).
    return np.where(r <= R, k * Q / R, k * Q / np.maximum(r, R))


def perfil_campo(Q, R, r):
    """E(r) de una esfera conductora: 0 en r < R y kQ/r² en r >= R (V/m)."""
    r = np.asarray(r, dtype=float)
    return np.where(r < R, 0.0, k * Q / np.maximum(r, R) ** 2)


def calcular_trabajo_traslado(q0, Q, R, r_i, r_f):
    """Trabajo de la fuerza eléctrica y cambio de energía potencial de q0.

    Mueve q0 de r_i a r_f en el campo de una esfera conductora de radio R y carga Q.
    Usa perfil_potencial, así que también es válida para puntos dentro del conductor.

    Returns:
        (W, delta_U) en julios, con W = -q0·ΔV y ΔU = q0·ΔV.
    """
    delta_V = perfil_potencial(Q, R, r_f) - perfil_potencial(Q, R, r_i)
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


# =====================================================================
# 5. PRUEBAS INDEPENDIENTES (cada una devuelve (aprobada, detalle))
# =====================================================================
def prueba_V_igual_Q_sobre_C():
    """V = Q/C. La discrepancia debe ser EXACTAMENTE la que predicen las constantes.

    V_Q/C = V_fórmula / (k·4π·ε0), así que la diferencia relativa predicha es
    |1/(k·4π·ε0) - 1|. Cubre calcular_capacitancia_esfera y calcular_potencial_superficie.
    """
    V_QC = Q / calcular_capacitancia_esfera(R_esfera)
    V_formula = calcular_potencial_superficie(Q, R_esfera)
    medida = diferencia_relativa(V_QC, V_formula)
    predicha = abs(1 / PRODUCTO_CONSTANTES - 1)
    ok = abs(medida - predicha) <= 1e-9
    return ok, f"dif. relativa medida = {medida:.4e}, predicha por las constantes = {predicha:.4e}"


def prueba_campo_afuera_igual_sigma_sobre_eps0():
    """E justo afuera = σ/ε0 con σ = Q/(4πR²). Usa ε0 (la fórmula del código usa k).

    Cubre calcular_campo_superficie. Tolerancia: la de las constantes.
    """
    E_gauss = Q / (4 * np.pi * R_esfera ** 2) / eps_0
    E_codigo = calcular_campo_superficie(Q, R_esfera)
    dif = diferencia_relativa(E_codigo, E_gauss)
    return dif <= TOL_CONSTANTES, f"E código = {E_codigo:.3f} V/m, σ/ε0 = {E_gauss:.3f} V/m, dif. = {dif:.3e}"


def prueba_capacitor_desde_integral_de_E():
    """C = Q/ΔV con ΔV = ∫E dr entre a y b, para κ = 1 y κ = 2 (E se reduce a E/κ).

    Cubre perfil_campo y capacitancia_capacitor_esferico (incluido el papel de κ).
    Tolerancia: la de las constantes (la integral usa k; la fórmula usa ε0).
    """
    a, b = R_esfera, R_coraza_interna
    ok, partes = True, []
    for kap in (1.0, kappa):
        delta_V, _ = quad(lambda r: float(perfil_campo(Q, a, r)) / kap, a, b)
        C_integral = Q / delta_V
        C_formula = capacitancia_capacitor_esferico(a, b, kap)
        dif = diferencia_relativa(C_formula, C_integral)
        ok = ok and dif <= TOL_CONSTANTES
        partes.append(f"κ={kap:g}: C integral = {C_integral * 1e12:.4f} pF, "
                      f"fórmula = {C_formula * 1e12:.4f} pF, dif. = {dif:.2e}")
    return ok, "; ".join(partes)


def prueba_E_igual_menos_dV_dr():
    """E = -dV/dr con derivada numérica, lejos del salto de la superficie.

    La diferencia central de 1/r tiene error relativo (h/r)², así que la tolerancia es
    1.5·(h/r_min)² (50 % de margen). Cubre perfil_potencial y perfil_campo.
    """
    r = np.linspace(1.2 * R_esfera, 3.0 * R_esfera, 2001)
    h = r[1] - r[0]
    E_numerico = -np.gradient(perfil_potencial(Q, R_esfera, r), r)
    E_modelo = perfil_campo(Q, R_esfera, r)
    error = np.max(np.abs(E_numerico[1:-1] - E_modelo[1:-1]) / E_modelo[1:-1])
    tolerancia = 1.5 * (h / r[1]) ** 2
    return error <= tolerancia, f"error relativo máximo = {error:.2e}, tolerancia = {tolerancia:.2e}"


def prueba_derivada_en_la_superficie():
    """En r = R la derivada numérica da ≈ la mitad del salto de E (no el valor de afuera).

    El cociente central mezcla un punto interior (V constante) y uno exterior. Se compara
    con la predicción exacta del cociente y se verifica que está cerca de la mitad del salto.
    """
    R = R_esfera
    r = np.linspace(0.9 * R, 1.1 * R, 101)
    i = int(np.argmin(np.abs(r - R)))
    E_en_R = -np.gradient(perfil_potencial(Q, R, r), r)[i]
    prediccion = (k * Q / R - k * Q / r[i + 1]) / (r[i + 1] - r[i - 1])
    mitad_salto = k * Q / R ** 2 / 2
    dif_pred = diferencia_relativa(E_en_R, prediccion)
    dif_mitad = diferencia_relativa(E_en_R, mitad_salto)
    h = r[i + 1] - r[i]
    ok = dif_pred <= 1e-9 and dif_mitad <= 1.5 * h / R
    return ok, (f"E_num(R) = {E_en_R:.3f} V/m, mitad del salto = {mitad_salto:.3f} V/m "
                f"(dif. {dif_mitad:.2e}), vs predicción del cociente: {dif_pred:.1e}")


def prueba_trabajo_contra_integral():
    """W = ∫q0·E dr comparado con calcular_trabajo_traslado, en tres recorridos.

    Recorridos: R→2R, 0.5R→2R (cruza la superficie; adentro E = 0) y 2R→R (regreso, W
    cambia de signo). Solo interviene k, así que la tolerancia es de 1e-9.
    Cubre perfil_campo y calcular_trabajo_traslado.
    """
    R = R_esfera
    W_ida, _ = quad(lambda r: q0 * float(perfil_campo(Q, R, r)), R, 2 * R)
    casos = [("R→2R", R, 2 * R, W_ida),
             ("0.5R→2R", 0.5 * R, 2 * R, W_ida),
             ("2R→R", 2 * R, R, -W_ida)]
    ok, partes = True, []
    for nombre, r_i, r_f, esperado in casos:
        W, delta_U = calcular_trabajo_traslado(q0, Q, R, r_i, r_f)
        dif = diferencia_relativa(W, esperado)
        ok = ok and dif <= 1e-9 and abs(delta_U + W) <= 1e-9 * abs(W)
        partes.append(f"{nombre}: W = {W * 1e9:.3f} nJ (dif. {dif:.1e})")
    return ok, "; ".join(partes)


def prueba_energia_tres_formulas():
    """U = ½CV² (código) frente a ½QV (no usa C) y a Q²/(2C) (no usa V).

    Cada comparación mezcla k y ε0 una vez, así que la tolerancia es la de las constantes.
    Cubre calcular_energia_esfera.
    """
    C = calcular_capacitancia_esfera(R_esfera)
    V = calcular_potencial_superficie(Q, R_esfera)
    U_codigo = calcular_energia_esfera(C, V)
    U_QV = 0.5 * Q * V
    U_Q2C = Q ** 2 / (2 * C)
    d1 = diferencia_relativa(U_codigo, U_QV)
    d2 = diferencia_relativa(U_Q2C, U_QV)
    ok = d1 <= TOL_CONSTANTES and d2 <= TOL_CONSTANTES
    return ok, (f"U código = {U_codigo * 1e9:.4f} nJ, ½QV = {U_QV * 1e9:.4f} nJ, "
                f"Q²/2C = {U_Q2C * 1e9:.4f} nJ (dif. {d1:.2e} y {d2:.2e})")


def prueba_estado_capacitor():
    """Q y U del capacitor a 100 V, comparados con Gauss y con fórmulas equivalentes.

    Con ΔV fija, E(a) = ΔV/(a²(1/a - 1/b)) y Gauss da Q = κ·ε0·E(a)·4πa² (usa solo ε0,
    como la fórmula de C). Verifica κ = 1 y κ = 2. Tolerancia: 1e-12 (solo redondeo).
    Cubre calcular_estado_capacitor y capacitancia_capacitor_esferico.
    """
    a, b = R_esfera, R_coraza_interna
    ok, partes = True, []
    for kap in (1.0, kappa):
        C = capacitancia_capacitor_esferico(a, b, kap)
        Q_c, U_c = calcular_estado_capacitor(C, V_capacitor)
        E_a = V_capacitor / (a * a * (1 / a - 1 / b))
        Q_gauss = kap * eps_0 * E_a * 4 * np.pi * a * a
        d_Q = diferencia_relativa(Q_c, Q_gauss)
        d_U1 = diferencia_relativa(U_c, 0.5 * Q_c * V_capacitor)
        d_U2 = diferencia_relativa(U_c, Q_c ** 2 / (2 * C))
        ok = ok and max(d_Q, d_U1, d_U2) <= 1e-12
        partes.append(f"κ={kap:g}: Q = {Q_c * 1e9:.4f} nC (Gauss dif. {d_Q:.1e}), "
                      f"U = {U_c * 1e9:.4f} nJ (dif. {d_U1:.1e}, {d_U2:.1e})")
    return ok, "; ".join(partes)


def prueba_leyes_de_escala_del_barrido():
    """Los exponentes de R en el barrido deben ser enteros exactos (leyes de potencias).

    Q fija: C ∝ R, V ∝ 1/R, E ∝ 1/R², U ∝ 1/R. V fijo: Q ∝ R, E ∝ 1/R, U ∝ R.
    Tolerancia 1e-9 sobre cada exponente.
    """
    C = calcular_capacitancia_esfera(R_barrido)
    V = calcular_potencial_superficie(Q, R_barrido)
    V_fijo = calcular_potencial_superficie(Q, R_esfera)
    Q_b = C * V_fijo
    esperados = [("C", C, 1), ("V(Q fija)", V, -1),
                 ("E(Q fija)", calcular_campo_superficie(Q, R_barrido), -2),
                 ("U(Q fija)", calcular_energia_esfera(C, V), -1),
                 ("Q(V fijo)", Q_b, 1),
                 ("E(V fijo)", calcular_campo_superficie(Q_b, R_barrido), -1),
                 ("U(V fijo)", calcular_energia_esfera(C, V_fijo), 1)]
    ok, partes = True, []
    for nombre, y, exponente in esperados:
        medido = pendiente_loglog(R_barrido, y)
        ok = ok and abs(medido - exponente) <= 1e-9
        partes.append(f"{nombre}: {medido:+.6f}")
    return ok, ", ".join(partes)


def prueba_convergencia_capacitor():
    """Para b >> a, (C_capacitor - C_esfera)/C_esfera = a/(b - a) → 0 como a/b.

    Identidad exacta: C_capacitor = C_esfera·b/(b - a). Tolerancia 1e-6 sobre la
    desviación relativa. Cubre capacitancia_capacitor_esferico.
    """
    a = R_esfera
    cocientes = np.array([10.0, 100.0, 1000.0, 10000.0])
    C_aislada = calcular_capacitancia_esfera(a)
    desviacion = (capacitancia_capacitor_esferico(a, a * cocientes) - C_aislada) / C_aislada
    prediccion = 1 / (cocientes - 1)
    error = np.max(np.abs(desviacion - prediccion) / prediccion)
    return error <= 1e-6, ("desviaciones para b/a = 10, 1e2, 1e3, 1e4: "
                           + ", ".join(f"{d:.2e}" for d in desviacion))


PRUEBAS = [
    ("V = Q/C", prueba_V_igual_Q_sobre_C),
    ("E afuera = σ/ε0", prueba_campo_afuera_igual_sigma_sobre_eps0),
    ("C desde ∫E (κ = 1 y 2)", prueba_capacitor_desde_integral_de_E),
    ("E = -dV/dr lejos del salto", prueba_E_igual_menos_dV_dr),
    ("derivada numérica en r = R", prueba_derivada_en_la_superficie),
    ("W contra ∫q0·E", prueba_trabajo_contra_integral),
    ("U por tres fórmulas", prueba_energia_tres_formulas),
    ("estado del capacitor (Gauss)", prueba_estado_capacitor),
    ("leyes de escala del barrido", prueba_leyes_de_escala_del_barrido),
    ("convergencia b → ∞", prueba_convergencia_capacitor),
]


def ejecutar_pruebas(mostrar=True):
    """Ejecuta todas las pruebas y devuelve {nombre: (aprobada, detalle)}."""
    resultados = {}
    for nombre, prueba in PRUEBAS:
        try:
            ok, detalle = prueba()
        except Exception as error:  # una función rota puede lanzar excepciones
            ok, detalle = False, f"excepción: {error!r}"
        resultados[nombre] = (bool(ok), detalle)
        if mostrar:
            print(f"[{'APROBADA' if ok else 'FALLIDA '}] {nombre}\n           {detalle}")
    return resultados


# =====================================================================
# 6. PRUEBAS DE MUTACIÓN: se rompe a propósito cada función y se verifica
#    que al menos una prueba lo detecte
# =====================================================================
def _trabajo_signo_invertido(q, Q_, R, r_i, r_f):
    delta_V = perfil_potencial(Q_, R, r_f) - perfil_potencial(Q_, R, r_i)
    return float(q * delta_V), float(q * delta_V)  # W con el signo equivocado


def _mutaciones():
    """Lista de (función, descripción del error, versión rota con la misma firma)."""
    return [
        ("calcular_capacitancia_esfera", "usa 2π en vez de 4π",
         lambda R: 2 * np.pi * eps_0 * R),
        ("calcular_potencial_superficie", "divide entre R² en vez de R",
         lambda Q_, R: k * Q_ / (R * R)),
        ("calcular_campo_superficie", "divide entre R en vez de R²",
         lambda Q_, R: k * Q_ / R),
        ("calcular_energia_esfera", "olvida el factor ½",
         lambda C, V: C * V * V),
        ("perfil_potencial", "decae como 1/r² afuera",
         lambda Q_, R, r: np.where(np.asarray(r) <= R, k * Q_ / R,
                                   k * Q_ / np.maximum(np.asarray(r), R) ** 2)),
        ("perfil_campo", "decae como 1/r afuera",
         lambda Q_, R, r: np.where(np.asarray(r) < R, 0.0,
                                   k * Q_ / np.maximum(np.asarray(r), R))),
        ("calcular_trabajo_traslado", "signo de W invertido", _trabajo_signo_invertido),
        ("capacitancia_capacitor_esferico", "denominador b + a en vez de b - a",
         lambda a, b, kap=1.0: kap * 4 * np.pi * eps_0 * (a * b / (b + a))),
        ("capacitancia_capacitor_esferico", "ignora κ",
         lambda a, b, kap=1.0: 4 * np.pi * eps_0 * (a * b / (b - a))),
        ("calcular_estado_capacitor", "U sin el factor ½",
         lambda C, V: (C * V, C * V * V)),
        ("calcular_estado_capacitor", "Q = C/V en vez de C·V",
         lambda C, V: (C / V, 0.5 * C * V * V)),
    ]


def ejecutar_mutaciones():
    """Sustituye cada función por una versión rota, corre las pruebas y restaura."""
    print("\n=== Verificación por mutación: ¿alguna prueba detecta cada error? ===")
    todas = True
    for nombre_funcion, descripcion, version_rota in _mutaciones():
        original = globals()[nombre_funcion]
        globals()[nombre_funcion] = version_rota
        try:
            resultados = ejecutar_pruebas(mostrar=False)
        finally:
            globals()[nombre_funcion] = original
        detectoras = [n for n, (ok, _) in resultados.items() if not ok]
        todas = todas and bool(detectoras)
        estado = "DETECTADO" if detectoras else "NO DETECTADO"
        print(f"{estado:>13} | {nombre_funcion}: {descripcion}")
        print(f"{'':>13} |   falla en: {', '.join(detectoras) if detectoras else '-'}")
    print("Todas las mutaciones fueron detectadas." if todas
          else "ATENCIÓN: hay errores que ninguna prueba detecta.")
    return todas


# =====================================================================
# 7. PROGRAMA PRINCIPAL
# =====================================================================
def main():
    """Ejecuta las actividades, las pruebas independientes y la verificación por mutación."""
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
    aprobadas = sum(ok for ok, _ in resultados.values())
    print(f"\nPruebas aprobadas: {aprobadas}/{len(resultados)}")
    ejecutar_mutaciones()


if __name__ == "__main__":
    main()