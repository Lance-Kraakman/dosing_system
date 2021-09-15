import numpy as np
import control as ct
import matplotlib.pyplot as plt

PRINT_INFO = False
ADAPTIVE_CONTROL = True

SECONDS_IN_DAY = 10000
SECONDS_IN_HOUR = 3600

# Specify Data Points
TIME_LENGTH = SECONDS_IN_DAY # Sample Time n days (24 Hour Period)
TIME_PERIOD = 1 # One sample every n hours

# Creating Time Step Array
T = np.linspace(0, TIME_LENGTH, int(TIME_LENGTH / TIME_PERIOD) + 1)

print(T)
# C_leachate_H = params.get('C_leachate_H', np.float_power(10, -12.2))  # mol/L
# C_leachate_OH = params.get('C_leachate_OH', 0.01584893192)

# Initial Conditions
[x0, x1, x2, x3, x4] = [np.float_power(10, -12), 0.01, 0, 1.8, 0]

# Create Plant Noise
noise1 = 2 * np.random.normal(0.5, 2.5)
noise2 = abs(np.sin(0.1 * np.pi * T))
noise3 = 200 * abs(np.sin(1 * np.pi * T))
noise4 = 200 * abs(np.sin(1 * np.pi * T))
plant_noise = 100 * (noise4 + noise3)  # Summing all of our noises


acid_input = 0.0001

acid_input_arr = acid_input * np.ones(T.shape)

leachate_flow = 208 * np.ones(T.shape)  # leachate flow in L/S 2.5*1.157407
zeros = np.zeros(T.shape)  # acid flow system input (additional to controller acid flow)
pond_level = 1 * np.ones(T.shape)  # Pond Level Set as Input incase we need to add plant dynamics.


def ph_calc(t, x, u, params={}):
    # State Variables
    C_H = x[1]
    C_OH = x[0]
    b = (C_OH - C_H) # it shifts this in one direction ?

    C_star = (1 / 2) * (-b + np.sqrt(np.power(b, 2) + 4.0 * (10 ** -14.0)))
    if b == 0 or C_star == 0:
        PH = 0
    elif C_star < 0:
        PH = np.log10(C_star)
    else:
        PH = -1 * np.log10(C_star)

    PH = PH
    u1 = u[0]

    if PRINT_INFO:
        print("PH: " + PH.__str__() + " C_H : " + C_H.__str__() + " C_OH : " + C_OH.__str__())

    return [C_OH, C_H, PH, u1]


def dosing_plant(t, x, u, params={}):
    # Dosing Equation Parameters
    C_acid_H = params.get('C_acid_H', np.float_power(10, -0.3))
    C_acid_OH = params.get('C_acid_OH', 2*np.float_power(10, -14.0))
    pond_volume = params.get('p_volume', 1500)
    C_leachate_H = params.get('C_leachate_H', np.float_power(10, -12.2)) # mol/L
    C_leachate_OH = params.get('C_leachate_OH', 0.01584893192)

    # Map the states to local variables
    x1 = x[0]  # OH- Conc
    x2 = x[1]  # H+ Conc

    x3 = x[2] # V Pond Volume
    T = params.get('TIME_PERIOD', 1)

    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    u4 = u[3]
    # volume = x[2]

    acid_flow = u[0]  # Maximum Flow Rate of 7.5 L/S
    leachate_flow = u[1]
    plant_noise = u[3]

    # dx1 = (1 / u4) * (
    #         (leachate_flow * C_leachate_OH) - ((leachate_flow + acid_flow) * x1) + (acid_flow * C_acid_OH))
    # dx2 = (1 / u4) * (
    #         (leachate_flow * C_leachate_H) - ((leachate_flow + acid_flow) * x2) + (acid_flow * C_acid_H))

    dx1 = (1 / x3) * (
            (leachate_flow * C_leachate_OH) + (acid_flow * C_acid_OH))
    dx2 = (1 / x3) * (
            (leachate_flow * C_leachate_H) + (acid_flow * C_acid_H))

    dx3 = (leachate_flow + acid_flow)
    # dx3 = 0
    print(x3)



    # print("Volume: " + volume.__str__())

    if PRINT_INFO:
        print("Dosing Plant leachate  :" + u[1].__str__())
        print("Dosing Plant acid flow :" + u[0].__str__())
        print("Dosing Plant Noise :" + u[2].__str__())
        print("Dosing Plant leachate  :" + u[1].__str__())
        print("Dosing Plant acid flow :" + u[0].__str__())
        print("Dosing Plant Noise :" + u[2].__str__())
        print("OH dt: " + dx1.__str__() + " H dt: " + dx2.__str__())

    return [dx1, dx2, dx3]


# Define the IO system for the plant
io_dosing = ct.NonlinearIOSystem(dosing_plant, outfcn=ph_calc,
                                 inputs=('acid_flow', 'leachate_flow', 'noise', 'pond_volume'),
                                 outputs=['x1', 'x2', 'y', 'u1'],
                                 states=('x1', 'x2', 'x3'), name='io_dosing')

response = ct.input_output_response(io_dosing, T, [acid_input_arr, leachate_flow, plant_noise,
                                                                pond_level], [x1, x2, x3], return_x=True, params={'TIME_PERIOD': TIME_PERIOD})

time = response[0]
outputs = response[1]

fig1, axs = plt.subplots(2, 2)

axs[0, 0].plot(time, outputs[0], label='C_OH-')
axs[0, 0].plot(time, outputs[1], label='C_H+')
axs[0, 0].set_title("Pond Concentrations")
axs[0, 0].legend()

axs[0, 1].step(time, outputs[3], label='Acid Flow L/h')
axs[0, 1].step(time, leachate_flow, label='Leachate Flow L/h')
axs[0, 1].set_title("Acid Flows")
axs[0, 1].legend()



axs[1, 1].set_title("Flows and PHs =")
axs[1, 1].step(time, outputs[3], label='Acid Flow L/s', marker='*')
axs[1, 1].step(time, leachate_flow, label='Leachate Flow L/s')
# axs[1, 1].plot(time, setpoint_array, label='Set Point', linestyle='-')
axs[1, 1].plot(time, outputs[2], label='PH')
axs[1, 1].legend()

plt.show()