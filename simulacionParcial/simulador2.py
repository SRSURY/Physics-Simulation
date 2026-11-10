"""Proyecto 10: actividades 2, 3, 4, 7 y 8 de electrostática.

Calcula las propiedades de una esfera conductora y muestra gráficas
comparativas con carga fija y con potencial fijo.
"""

import numpy as np
import matplotlib.pyplot as plt


# Constantes eléctricas en unidades del Sistema Internacional.
k = 8.988e9  # Constante de Coulomb (N·m²/C²).
eps_0 = 8.854e-12  # Permitividad eléctrica del vacío (F/m).

# Datos oficiales del proyecto; las pruebas de la actividad 8 no los modifican.
R_esfera = 0.05  # Radio de referencia: 5 cm (m).
Q_esfera = 1.0e-9  # Carga de la esfera: 1 nC (C).
q0_prueba = 1.0e-9  # Carga de prueba oficial: 1 nC (C).

# Nueve radios para el barrido, incluidos 3, 5 y 8 cm (todos en metros).
radios = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
radios_destacados = [0.03, 0.05, 0.08]


def capacitancia_esfera(radio):
    """Calcula C = 4*pi*eps_0*R para una esfera aislada (F)."""
    return 4 * np.pi * eps_0 * radio


def potencial_superficie(carga, radio):
    """Calcula V = k*Q/R en la superficie de la esfera (V)."""
    return k * carga / radio


def campo_superficie(carga, radio):
    """Calcula E = k*Q/R**2 justo afuera de la esfera (V/m)."""
    return k * carga / radio**2


def energia_capacitor(capacitancia, voltaje):
    """Calcula U = 1/2*C*V**2, energía almacenada (J)."""
    return 0.5 * capacitancia * voltaje**2


def trabajo_fuerza_electrica(carga_prueba, carga_esfera, radio_inicial, radio_final):
    """Calcula W eléctrico y el cambio de energía q0*deltaV (J).

    Se usa para mover una carga entre puntos en el exterior de la esfera.
    """
    # El potencial en un punto exterior es V = k*Q/r.
    voltaje_inicial = k * carga_esfera / radio_inicial
    voltaje_final = k * carga_esfera / radio_final
    cambio_voltaje = voltaje_final - voltaje_inicial

    # El campo realiza W = -q0*deltaV; la energía potencial cambia en q0*deltaV.
    trabajo = -carga_prueba * cambio_voltaje
    cambio_energia_potencial = carga_prueba * cambio_voltaje
    return trabajo, cambio_energia_potencial


def marcar_radios_destacados(radios_cm, valores):
    """Destaca en una curva los puntos correspondientes a 3, 5 y 8 cm."""
    for radio_marcado in radios_destacados:
        indice = np.argmin(np.abs(radios_cm - radio_marcado * 100))
        # El punto coloreado marca el dato y la línea vertical señala su radio.
        plt.scatter(radios_cm[indice], valores[indice], color="red", zorder=3)
        plt.axvline(radio_marcado * 100, color="gray", linestyle=":", alpha=0.5)


def actividad_2_3_4():
    """Imprime y grafica C, V, E y U al cambiar R con Q constante."""
    # Se aplican las fórmulas de la esfera usando la misma carga para cada radio.
    capacitancias = capacitancia_esfera(radios)
    potenciales = potencial_superficie(Q_esfera, radios)
    campos = campo_superficie(Q_esfera, radios)
    energias = energia_capacitor(capacitancias, potenciales)

    print("\nActividades 2-4: barrido del radio con Q fija")
    print(f"Carga constante: Q = {Q_esfera * 1e9:.2f} nC")
    print(f"{'R (cm)':>8} {'C (pF)':>12} {'V (V)':>12} "
          f"{'E (kV/m)':>12} {'U (nJ)':>12}")

    for i, radio in enumerate(radios):
        # Las conversiones solo cambian la unidad mostrada, no los cálculos SI.
        print(f"{radio * 100:8.1f} "
              f"{capacitancias[i] * 1e12:12.3f} "
              f"{potenciales[i]:12.2f} "
              f"{campos[i] / 1e3:12.2f} "
              f"{energias[i] * 1e9:12.3f}")

    # Cuatro paneles muestran cómo depende cada magnitud del radio.
    radios_cm = radios * 100
    datos_graficas = [
        (capacitancias * 1e12, "Capacitancia C(R)", "C (pF)"),
        (potenciales, "Potencial V(R)", "V (V)"),
        (campos / 1e3, "Campo justo afuera E(R)", "E (kV/m)"),
        (energias * 1e9, "Energía almacenada U(R)", "U (nJ)"),
    ]

    plt.figure(figsize=(11, 8))
    for numero, (valores, titulo, etiqueta_y) in enumerate(datos_graficas, start=1):
        plt.subplot(2, 2, numero)
        plt.plot(radios_cm, valores, "o-", label="Q fija")
        marcar_radios_destacados(radios_cm, valores)
        plt.title(titulo)
        plt.xlabel("Radio R (cm)")
        plt.ylabel(etiqueta_y)
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.suptitle("Actividades 2-4: esfera conductora con Q = 1 nC")
    plt.tight_layout()
    plt.savefig("barrido_R_Q_fija.png", dpi=300)
    plt.show()


def actividad_7():
    """Compara resultados con carga fija y con potencial fijo."""
    # El potencial constante se toma del caso oficial de R = 5 cm y Q = 1 nC.
    voltaje_fijo = potencial_superficie(Q_esfera, R_esfera)

    # Caso 1: carga fija; se recalculan V, E y U para cada radio.
    C_carga_fija = capacitancia_esfera(radios)
    Q_carga_fija = np.full(radios.shape, Q_esfera)
    V_carga_fija = potencial_superficie(Q_carga_fija, radios)
    E_carga_fija = campo_superficie(Q_carga_fija, radios)
    U_carga_fija = energia_capacitor(C_carga_fija, V_carga_fija)

    # Caso 2: potencial fijo; Q = C*V, después se calculan E y U.
    C_voltaje_fijo = C_carga_fija
    Q_voltaje_fijo = C_voltaje_fijo * voltaje_fijo
    V_voltaje_fijo = np.full(radios.shape, voltaje_fijo)
    E_voltaje_fijo = campo_superficie(Q_voltaje_fijo, radios)
    U_voltaje_fijo = energia_capacitor(C_voltaje_fijo, V_voltaje_fijo)

    print("\nActividad 7: barrido con potencial fijo")
    print(f"Potencial fijo: V = {voltaje_fijo:.2f} V, tomado de R = 5 cm")
    print(f"{'R (cm)':>8} {'C (pF)':>12} {'Q (nC)':>12} "
          f"{'V (V)':>12} {'E (kV/m)':>12} {'U (nJ)':>12}")

    for i, radio in enumerate(radios):
        print(f"{radio * 100:8.1f} "
              f"{C_voltaje_fijo[i] * 1e12:12.3f} "
              f"{Q_voltaje_fijo[i] * 1e9:12.3f} "
              f"{V_voltaje_fijo[i]:12.2f} "
              f"{E_voltaje_fijo[i] / 1e3:12.2f} "
              f"{U_voltaje_fijo[i] * 1e9:12.3f}")

    # La figura permite comparar cómo cambian Q, V, E y U bajo ambas condiciones.
    comparaciones = [
        (Q_carga_fija * 1e9, Q_voltaje_fijo * 1e9, "Carga Q (nC)"),
        (V_carga_fija, V_voltaje_fijo, "Potencial V (V)"),
        (E_carga_fija / 1e3, E_voltaje_fijo / 1e3, "Campo E (kV/m)"),
        (U_carga_fija * 1e9, U_voltaje_fijo * 1e9, "Energía U (nJ)"),
    ]

    plt.figure(figsize=(11, 8))
    for numero, (valores_Q, valores_V, etiqueta_y) in enumerate(comparaciones, start=1):
        plt.subplot(2, 2, numero)
        plt.plot(radios * 100, valores_Q, "o-", label="Q fija")
        plt.plot(radios * 100, valores_V, "s-", label="V fijo")
        plt.axvline(R_esfera * 100, color="gray", linestyle=":", alpha=0.6)
        plt.title(etiqueta_y + " en función del radio")
        plt.xlabel("Radio R (cm)")
        plt.ylabel(etiqueta_y)
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.suptitle("Actividad 7: comparación de Q fija y V fijo")
    plt.tight_layout()
    plt.savefig("actividad_7_comparacion.png", dpi=300)
    plt.show()

    # En R = 5 cm se eligió V fijo igual al potencial del caso de Q fija,
    # por lo que ambos casos tienen la misma Q, V, E y U en ese punto.
    print("Las condiciones coinciden en R = 5 cm porque allí se definió el "
          "potencial fijo usando Q = 1 nC.")


def actividad_8():
    """Compara el trabajo eléctrico con la energía de la esfera."""
    capacitancia = capacitancia_esfera(R_esfera)
    voltaje = potencial_superficie(Q_esfera, R_esfera)
    energia_esfera = energia_capacitor(capacitancia, voltaje)
    producto_constantes = k * 4 * np.pi * eps_0

    # Casos de prueba separados: no modifican q0_prueba ni Q_esfera oficiales.
    experimentos = [
        ("Datos oficiales: q0 = Q y destino 2R",
         q0_prueba, 2 * R_esfera),
        ("Experimento: q0 = 2 nC y destino 2R",
         2.0e-9, 2 * R_esfera),
        ("Experimento: q0 oficial y destino 3R",
         q0_prueba, 3 * R_esfera),
    ]

    print("\nActividad 8: trabajo de la fuerza eléctrica y energía de la esfera")
    print("W_eléctrico = q0 * [V(R) - V(r_final)]")
    print("U_esfera = 1/2 * C * V(R)**2")
    print(f"k * 4*pi*eps_0 = {producto_constantes:.9f}")

    for nombre, carga_prueba, radio_final in experimentos:
        # Se anticipa W/U antes de calcular el trabajo de cada experimento.
        razon_esperada = (
            2 * (carga_prueba / Q_esfera)
            * (1 - R_esfera / radio_final)
            / producto_constantes
        )
        print(f"\n{nombre}")
        print(f"Razón W/U esperada: {razon_esperada:.6f}")

        # W es el trabajo realizado por el campo eléctrico sobre q0.
        trabajo, cambio_energia = trabajo_fuerza_electrica(
            carga_prueba, Q_esfera, R_esfera, radio_final
        )
        razon_calculada = trabajo / energia_esfera
        diferencia_relativa = (
            abs(razon_calculada - razon_esperada) / abs(razon_esperada)
        )

        print(f"Trabajo de la fuerza eléctrica W: {trabajo * 1e9:.3f} nJ")
        print(f"Cambio de energía potencial ΔU: {cambio_energia * 1e9:.3f} nJ")
        print(f"Energía almacenada en la esfera U: {energia_esfera * 1e9:.3f} nJ")
        print(f"Razón W/U calculada: {razon_calculada:.6f}")
        print(f"Diferencia relativa: {diferencia_relativa:.3e}")

        # Para una carga positiva que se aleja, el campo hace W positivo
        # y la energía potencial disminuye (ΔU es negativo).
        if trabajo > 0 and cambio_energia < 0:
            print("Interpretación: W > 0 y ΔU < 0; el campo realiza trabajo "
                  "y la energía potencial disminuye.")

    print("\nConclusión: para q0 = Q y destino 2R, la diferencia de potencial "
          "es V(R)/2 y W/U queda aproximadamente en 1. Si cambia q0, el trabajo "
          "cambia en proporción a q0; si cambia el destino, cambia la diferencia "
          "de potencial. Por eso la razón deja de ser aproximadamente 1.")


def main():
    """Ejecuta en orden las actividades solicitadas."""
    print("Proyecto 10: esfera conductora")
    actividad_2_3_4()
    actividad_7()
    actividad_8()


if __name__ == "__main__":
    main()
