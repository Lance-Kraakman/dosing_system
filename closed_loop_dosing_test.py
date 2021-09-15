# Dosing system
import numpy as np
import control as ct
import matplotlib as mpl
import matplotlib.pyplot as plt

import math


def ph_calc(t, x, u, params={}):
    # State Variables
    C_H = x[1]
    C_OH = x[0]
    b = C_OH - C_H

    C_star = (1/2) * (-b + np.sqrt(abs(np.power(b, 2) + 4.0 * (10 ** -14.0))))
    if b == 0 or C_star == 0:
        PH = 0
    elif C_star < 0:
        PH = -1 * np.log10(C_star)
    else:
        PH = -1 * np.log10(C_star)

    PH = PH
    u1 = u[0]

    print("PH: " + PH.__str__() + " C_H : " + C_H.__str__() + " C_OH : " + C_OH.__str__())


    return [C_OH, C_H, PH, u1]


def dosing_plant(t, x, u, params={}):
    # Dosing Equation Parameters

    C_acid_H = params.get('C_acid_H', 0.99)
    C_acid_OH = params.get('C_acid_OH', 0.001)
    pond_volume = params.get('p_volume', 100000)

    C_leachate_H = params.get('C_leachate_H', 0.001)
    C_leachate_OH = params.get('C_leachate_OH', 0.99)

    # Map the states to local variables
    x1 = x[0]  # OH- Conc
    x2 = x[1]  # H+ Conc

    print("Dosing Plant leachate  :" + u[1].__str__())
    print("Dosing Plant acid flow :" + u[0].__str__())
    print("Dosing Plant u[2] :" + u[2].__str__())

    u1 = u[0]
    u2 = u[1]
    u3 = u[2]

    acid_flow = u[0] # Maximum Flow Rate of 7.5 L/S
    leachate_flow = u[1]



    # Compute the control action -> We are only allowed to add acid so acid must be positive
    # u[0] is acid and u[1] is our system noise

    dx1 = (1 / pond_volume) * (
            (leachate_flow * C_leachate_OH) - ((leachate_flow + acid_flow) * x1) + (acid_flow * C_acid_OH))
    dx2 = (1 / pond_volume) * (
            (leachate_flow * C_leachate_H) - ((leachate_flow + acid_flow) * x2) + (acid_flow * C_acid_H))

    # print("OH dt: " + dx1.__str__() + " H dt: " + dx2.__str__())
    # print(dx2.__str__() + "\n")

    print("x1 : dx1 = " + x1.__str__() + " : " + dx1.__str__())

    # Return
    return [dx1, dx2]


# Define the IO system for the plant
io_dosing = ct.NonlinearIOSystem(dosing_plant, outfcn=ph_calc, inputs=('acid_flow', 'leachate_flow', 'noise'),
                                 outputs=['x1', 'x2', 'y', 'u1'],
                                 states=('x1', 'x2', 'u1'), name='io_dosing')


# Create Generic PID Controller for acid input
Kp = 0.01
Ki = 0.0000000001

# Transfer function for acid dosing controller
control_tf = ct.tf2io(ct.TransferFunction([Kp, Ki], [1, 0.01 * Ki / Kp]), name='control', inputs='u', outputs='y')


# Functoin for custom acid dosing controller - Needs a state variable to track error accumulations
def cstm_dosing_ctrl(t, x, u, params={}):


    # Calculate the current error
    e = -1*(u[0])
    print("Error : " + e.__str__())

    # Calculate Integral error accumulation
    d_xI = (t * e)

    # Integral state accumulation dx_I/dt
    return d_xI


# Output function for controller
def cstm_ctrl_outfcn(t, x, u, params={}):

    # This should be a function of the PH Setpoint, PH error and Pond Volume
    Kp_Scale = 1
    Ki_Scale = 1

    Kp = 0.02
    Ki = 0.00001

    xI = x[0]
    e = -1 * (u[0])

    pid_value = (Kp_Scale * Kp * e) + (Ki_Scale * Ki * xI)

    # This is the one baby!
    y = np.clip(pid_value, 0, 7.5)

    print("PID Acid Dosing Value :" + y.__str__())
    print("PID Integral State Accum :" + xI.__str__())

    return [y]

# Transfer Function for gain scheduling
cust_ctrl = ct.NonlinearIOSystem(cstm_dosing_ctrl, outfcn=cstm_ctrl_outfcn, name='cust_ctrl', inputs=('u'),
                                 outputs=['y'], states=('xI'))

# Create our interconnected IO system
controlled_dosing = ct.InterconnectedSystem((control_tf, io_dosing), name='controlled_dosing',
                                            connections=(
                                                ['control.u', '-io_dosing.y'], ['io_dosing.acid_flow', 'control.y']),
                                            inplist=('control.u', 'io_dosing.leachate_flow', 'io_dosing.noise'),
                                            inputs=('vref', 'leachate_flow', 'noise'),
                                            outlist=['io_dosing.x1', 'io_dosing.x2', 'io_dosing.y', 'io_dosing.acid_flow'],
                                            outputs=['x1', 'x2', 'y', 'u1'],
                                            states=('x1', 'x2', 'y'))

# Create our inter connected IO system for our custom control function
cstm_ctrld_dosing = ct.InterconnectedSystem((cust_ctrl, io_dosing), name='cstm_ctrld_dosing', connections=(
    ['cust_ctrl.u', '-io_dosing.y'], ['io_dosing.acid_flow', 'cust_ctrl.y']), inplist=('cust_ctrl.u', 'io_dosing.leachate_flow', 'io_dosing.noise'),
            inputs=('vref', 'leachate_flow', 'noise'), outlist=['io_dosing.x1', 'io_dosing.x2', 'io_dosing.y', 'io_dosing.acid_flow'], states=('x1', 'x2', 'y'),
                    outputs=['x1', 'x2', 'y', 'u1'])

# Creating our data for an open loop simulation
T = np.linspace(0, 1000, 50)
print(T)

# Creating noise for our acid dosing readings
# noise 1 is at fs and is measurement error noise
noise1 = 2*np.random.normal(0.5, 2.5)
# noise 2 is a sinewave noise which is to simulate
noise2 = abs(np.sin(0.1*np.pi*T))
noise3 = 200*abs(np.sin(1*np.pi*T))
noise4 = 200*abs(np.sin(1*np.pi*T))

noise = 100*(noise4 + noise3) # Summing all of our noises
print(noise)

y_ref = 11 * np.ones(T.shape)
l_flow = np.zeros(T.shape)
zeros = np.zeros(T.shape)

# time, outputs = ct.input_output_response(controlled_dosing, T, [y_ref, l_flow, noise], [0.003, 0.002, 0])
time, outputs = ct.input_output_response(cstm_ctrld_dosing, T, [y_ref, l_flow, noise], [0.01, 0.01, 0])


plt.rcParamsDefault['lines.marker'] = '^'
plt.rcParamsDefault['lines.markersize'] = 10
# plt.rcParams['lines.linewidth'] = '3'
# plt.rcParams['lines.linestyle'] = '-'

fig1, axs = plt.subplots(3)

axs[0].plot(time, outputs[0], label='C_OH-')
axs[0].plot(time, outputs[1], label='C_H+')
axs[0].set_title("Pond Concentrations")
axs[0].legend()

axs[1].plot(time, outputs[2], label='PH')
axs[1].plot(time, outputs[2]-y_ref, label='PH Error')
axs[1].plot(time, y_ref, label='Set Point', linestyle='-')

axs[1].set_title("Pond PH")
axs[1].legend()

axs[2].plot(time, outputs[3], label='Acid Flow L/h')
axs[2].plot(time, l_flow, label='Leachate Flow L/h')
axs[2].set_title("Acid Flows")
axs[2].legend()

plt.show()



