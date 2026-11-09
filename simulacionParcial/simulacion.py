import numpy as np 
import matplotlib.pyplot as plt 

k = 8.988e9 ## Ley de Coulomb
eps_0 = 8.854e-12 ## Permitividad electrica del vacio (F/M)
R_inicial = 0.05 ## M   radio inicial de 5cm
Q = 1.0e-9 ## C (1 nC) Carga de la esfera 
q0 = 1.0e-9 ##  C (1 nC) Carga de prueba 
R_comparar = [0.03, 0.05, 0.08] ## Radios a comparar (3,5,8) cm

R_externo = 0.08 ## M   Coraza interna PROBLEMA 8
v_cap = 100.0 ## V   Voltaje para el capacitor esferico PROBLEMA 11
kappa = 2.0 ## Constante dielectrica PROBLEMA 12 

def calcular_capacitancia_esfera(R):
    C = 4*np.pi*eps_0*R
    return C ## Retorna la capacitancia

def calcular_potencial_superficie(Q, R):
    Vr = k*Q / R
    return Vr ## Retorna el voltaje en la superficie

def calcular_campo_superficie(Q,R):
    Er = k*Q / (R * R)
    return Er ## Retorna el campo electrico

def calcular_energia_esfera(C, V):
    U = 0.5*C*(V*V)
    return U ## Retorna la energia almacennada 

def perfil_potencial(r, R, Q):
    V_superficie = k * Q / R
    # Si r <= R toma el potencial de la superficie, si r > R cae como k*Q/r
    return np.where(r <= R, V_superficie, k * Q / r)

def perfil_campo(r, R, Q):
    # Si r < R el campo es 0, si r >= R cae como k*Q/r^2
    return np.where(r < R, 0.0, k * Q / (r * r))


def calcular_trabajo_traslado(q0, R_inicial, R_final, Q):
    V_i = k * Q / R_inicial
    V_f = k * Q / R_final
    delta_V = V_f - V_i

    W_elec = -q0 * delta_V
    delta_U = q0 * delta_V

    return W_elec, delta_U


def capacitancia_capacitor_esferico(R_interno, R_externo, kappa = 1.0):
    C = kappa*4*np.pi*eps_0*(R_interno*R_externo / (R_externo - R_interno))
    return C

def calcular_estado_capacitor(C, V_cap):
    Q = C * V_cap
    U = 0.5*C*(V_cap * V_cap)
    return Q, U

## For loop para que recorra la lista R_comparar 
# Actividades 2,3,4

for i in R_comparar:
    capacitancia = calcular_capacitancia_esfera(i)
    potencial = calcular_potencial_superficie(Q, i)
    campo = calcular_campo_superficie(Q,i)
    energia = calcular_energia_esfera(capacitancia, potencial)
    print(f"La Capacitancia es: {capacitancia}\nEl potencial de la esfera es: {potencial}\nEl campo electrico fuera de su superficie es: {campo}\nLa energia electrostactica es: {energia} \n")

voltaje_fijo = calcular_potencial_superficie(Q, R_inicial)

## Bucle para la actividad 7
for i in R_comparar:
    capacitancia = calcular_capacitancia_esfera(i)
    nueva_Carga = capacitancia * voltaje_fijo
    campo = calcular_campo_superficie(nueva_Carga,i)
    energia = calcular_energia_esfera(capacitancia, voltaje_fijo)
    print(f"La Capacitancia es: {capacitancia}\nEl potencial de la esfera es: {voltaje_fijo}\nEl campo electrico fuera de su superficie es: {campo}\nLa energia electrostactica es: {energia} \n")


## Actividad 8

trabajo_W, cambio_U = calcular_trabajo_traslado(q0, R_inicial, R_inicial*2, Q)
print(f"El trabajo de mover una carga de prueba desde R = 0.05m a 2R = 0.10m es: {trabajo_W:.3}")
print(f"El cambio de energía potencial es: {cambio_U:.3}")

## Actividad 5 Graficar V(r) y E(r)

r_vector = np.linspace(0.01, 0.20, 500)

perfilPotencial = perfil_potencial(r_vector, R_inicial, Q)
perfilCampo = perfil_campo(r_vector, R_inicial, Q)

## Perfil del Potencial Eléctrico V(r)

plt.figure(figsize=(8, 4))
# Argumentos de plt.plot: (Eje X en cm, Eje Y en Voltios, Estilo/Color, Etiqueta)
plt.plot(r_vector * 100, perfilPotencial, color='blue', linewidth=2, label='V(r)')
plt.axvline(x=R_inicial * 100, color='red', linestyle='--', label=f'Superficie R = {R_inicial*100:.0f} cm')

plt.title('Perfil de Potencial Eléctrico V(r) - Esfera Conductora')
plt.xlabel('Distancia radial r (cm)')
plt.ylabel('Potencial V (Voltios)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('grafica_potencial_V.png', dpi=300)
plt.show()


## Perfil del Campo Eléctrico E(r)

plt.figure(figsize=(8, 4))
# Argumentos de plt.plot: (Eje X en cm, Eje Y en kV/m [dividido entre 1000], Estilo/Color, Etiqueta)

plt.plot(r_vector * 100, perfilCampo / 1000, color='green', linewidth=2, label='E(r)')
plt.axvline(x=R_inicial * 100, color='red', linestyle='--', label=f'Superficie R = {R_inicial*100:.0f} cm')

plt.title('Perfil de Campo Eléctrico E(r) - Esfera Conductora')
plt.xlabel('Distancia radial r (cm)')
plt.ylabel('Campo Eléctrico E (kV/m)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('grafica_campo_E.png', dpi=300)
plt.show()


## Actividad 9 Superficies equipotenciales

# 1. Definir el plano espacial (de -20 cm a 20 cm en X e Y)
x = np.linspace(-0.20, 0.20, 400)
y = np.linspace(-0.20, 0.20, 400)
X, Y = np.meshgrid(x, y) # Crea la cuadrícula bidimensional

# 2. Calcular la distancia r al centro (0,0) para cada punto de la cuadrícula
r_plano = np.hypot(X, Y)

# 3. Calcular el voltaje en todo el plano usando tu función
# (perfil_potencial mantendrá el voltaje constante si r <= R_inicial)
V_plano = perfil_potencial(r_plano, R_inicial, Q)

# 4. Generar la gráfica
plt.figure(figsize=(7, 7))

# Dibuja las líneas equipotenciales (15 niveles de voltaje distintos)
contornos = plt.contour(X * 100, Y * 100, V_plano, levels=15, cmap='plasma')
plt.clabel(contornos, inline=True, fontsize=9, fmt='%.0f V') # Etiqueta los voltajes en las líneas

# Dibuja la esfera conductora sólida en el centro para dar contexto
esfera = plt.Circle((0, 0), R_inicial * 100, color='silver', ec='black', zorder=10, label='Esfera Conductora')
plt.gca().add_patch(esfera)

plt.title('Superficies Equipotenciales - Esfera Conductora (Vista 2D)')
plt.xlabel('Posición en X (cm)')
plt.ylabel('Posición en Y (cm)')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right')
plt.axis('equal') # Fuerza a que los ejes X e Y tengan la misma escala para que los círculos no se vean ovalados
plt.tight_layout()
plt.savefig('superficies_equipotenciales.png', dpi=300)
plt.show()





## Actividades 10, 11 

capacitancia_capacitor = capacitancia_capacitor_esferico(R_inicial, R_externo, 1)
estado_capacitorCarga, estado_capacitorEnergia = calcular_estado_capacitor(capacitancia_capacitor, v_cap)
print(f"La nueva capacitancia es: {capacitancia_capacitor}, la nueva carga es: {estado_capacitorCarga} y la nueva energia es: {estado_capacitorEnergia}")

## Actividad 12 
print("\nCambiamos el dielectrico a 2 para analizar el efecto que hace: ")
capacitancia_capacitor2 = capacitancia_capacitor_esferico(R_inicial, R_externo, kappa)
estado_capacitorCarga2, estado_capacitorEnergia2 = calcular_estado_capacitor(capacitancia_capacitor2, v_cap)
print(f"La nueva capacitancia es: {capacitancia_capacitor2}, la nueva carga es: {estado_capacitorCarga2} y la nueva energia es: {estado_capacitorEnergia2}")

