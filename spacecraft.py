import numpy as np
import matplotlib.pyplot as plt
import math

#givens
n_s = 32
n_p = 8
T_a = 298
R_th = 7.44
C_th = 483.57
R = 11
R_i = 15
C_i = 1.5
Tb_initial = 25
I_rr = 1000
sunlight_duration = 5400  #in seconds
eclipse_duration = 3000    #in seconds
T_ref = 25
alpha = 0.004
PPV = 50

V_i = 12
L_i = 20

V_ref = V_i
L_eq = L_i
R_eq = 5

v_bus = 12  #initial bus voltage
i_s = 2     #initial solar current

#time parameters
time_start = 0
time_end = 10
delta_t = 0.0001
time_array = np.arange(time_start, time_end, delta_t)

#define the derivative functions
def f(i_s, v_bus):
    return (1 / L_eq) * (V_ref - v_bus - R_eq * i_s)

def g(i_s, v_bus, P):
    return (1 / C_i) * (i_s - (v_bus / R_eq) - (P / v_bus))

#function to calculate the photovoltaic power provided by solar arrays
def calculate_PPV(Tb, solar_angle_degrees):
    G = 1361  #solar constant in W/m²
    solar_angle_radians = math.radians(solar_angle_degrees)  #convert angle to radians
    P_incident = G * math.cos(solar_angle_radians)  #adjust for solar angle
    eta_ref = 0.15  #reference efficiency
    beta = -0.005  #temperature coefficient
    eta_T = eta_ref * (1 + beta * (Tb - T_ref))  #efficiency adjusted for temperature
    PPV = eta_T * P_incident  #calculate PPV
    return PPV

#implementing the 4th order Runge-Kutta method
def runge_kutta_step(i_s, v_bus, delta_t, PPV):
    #calculate RK coefficients for i_s and v_bus in parallel
    k1_i = f(i_s, v_bus)
    k1_v = g(i_s, v_bus, PPV)

    k2_i = f(i_s + 0.5 * delta_t * k1_i, v_bus + 0.5 * delta_t * k1_v)
    k2_v = g(i_s + 0.5 * delta_t * k1_i, v_bus + 0.5 * delta_t * k1_v, PPV)

    k3_i = f(i_s + 0.5 * delta_t * k2_i, v_bus + 0.5 * delta_t * k2_v)
    k3_v = g(i_s + 0.5 * delta_t * k2_i, v_bus + 0.5 * delta_t * k2_v, PPV)

    k4_i = f(i_s + delta_t * k3_i, v_bus + delta_t * k3_v)
    k4_v = g(i_s + delta_t * k3_i, v_bus + delta_t * k3_v, PPV)

    i_s_new = i_s + (delta_t / 6) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
    v_bus_new = v_bus + (delta_t / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return i_s_new, v_bus_new

#initialize arrays to store the results
vbus_arr = []
is_arr = []
power_arr = []

desired_PL = 100
Tb = Tb_initial  #start with initial battery temperature

#time integration loop
for time in time_array:
    #vary solar angle (e.g., simulating sun movement)
    solar_angle = 45 + 45 * np.sin(np.pi * time / time_end)  #example sinusoidal variation
    PPV = calculate_PPV(Tb, solar_angle)  #calculate varying PPV

    #update current based on power losses
    P = v_bus * i_s - PPV  #update power difference
    if v_bus != 0:  # ensure v_bus is not zero
        i_s = desired_PL / v_bus  #maintain target power

    #update temperature based on power losses
    Tb += (P / C_th) * delta_t  #simplistic model for temperature increase
    
    #runge-Kutta step
    i_s, v_bus = runge_kutta_step(i_s, v_bus, delta_t, PPV)
    
    i_s = max(0, min(i_s, 10))  # adjust maximum current limit as needed
    
    print(f"Voltage: {v_bus}, Current: {i_s}")

    #check for NaNs or extremely large values and stop if found
    if np.isnan(i_s) or np.isnan(v_bus) or np.isinf(i_s) or np.isinf(v_bus):
        print(f"Unrealistic values encountered: i_s={i_s}, v_bus={v_bus}. Exiting loop.")
        break

    vbus_arr.append(v_bus)
    is_arr.append(i_s)
    power_arr.append(v_bus * i_s)

#print final results to debug
print(f'Final Bus Voltage: {v_bus:.2f} V')
print(f'Final Current: {i_s:.2f} A')

#identify Maximum Power Point (MPP)
if len(vbus_arr) > 0 and len(is_arr) > 0:
    vbus_arr = np.array(vbus_arr)
    is_arr = np.array(is_arr)
    mpp_index = np.argmax(vbus_arr * is_arr)  #MPP index
    mpp_voltage = vbus_arr[mpp_index]         #MPP voltage
    mpp_current = is_arr[mpp_index]           #MPP current
    print(f'Maximum Power Point: V = {mpp_voltage:.2f} V, I = {mpp_current:.2f} A')
else:
    print("No valid data points were generated.")

#plot the IV curve
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(vbus_arr, is_arr, label="i_s vs Vbus")
plt.xlabel('Vbus (Voltage)')
plt.ylabel('i_s (Current)')
plt.title('IV Curve of Simplified Spacecraft Power System')
plt.axvline(x=mpp_voltage, color='red', linestyle='--', label='MPP Voltage')
plt.axhline(y=mpp_current, color='blue', linestyle='--', label='MPP Current')
plt.legend()

#plot power as a function of bus voltage
plt.subplot(2, 1, 2)
plt.plot(vbus_arr, power_arr, label="Power vs Time")
plt.xlabel('Bus Voltage (V)')
plt.ylabel('Power (W)')
plt.title('Power Output of Simplified Spacecraft Power System')
plt.legend()

plt.tight_layout()
plt.show()
