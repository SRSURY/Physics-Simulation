import math
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



prueba = calcular_capacitancia_esfera(R_inicial)
print(prueba)








