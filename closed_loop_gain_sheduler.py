import numpy as np
import control as ct
import matplotlib.pyplot as plt

# ----------------              SIMULATION CONFIG               ------------
SECONDS_IN_DAY = 28800
SECONDS_IN_HOUR = 3600

# Print info during simulation
PRINT_INFO = False
ADAPTIVE_CONTROL = True

# Specify Data Point



TIME_LENGTH = 1000*1 #3*SECONDS_IN_DAY  # Sample Time n days (24 Hour Period)
TIME_PERIOD = 1  # One sample every n hours

# Creating Time Step Array
T = np.linspace(0, TIME_LENGTH, int(TIME_LENGTH / TIME_PERIOD) + 1)

# Initial Conditions
[x0, x1, x2, x3, x4] = [0, 0, np.float_power(10, -7), np.float_power(10, -7), 1288000]

# Custom Adaptive Controller Gains
KP = 1  # 8
KI = 0.1  # 9

# Create Plant Noise
noise1 = 2 * np.random.normal(0.5, 2.5)
noise2 = abs(np.sin(0.1 * np.pi * T))
noise3 = 200 * abs(np.sin(1 * np.pi * T))
noise4 = 200 * abs(np.sin(1 * np.pi * T))
plant_noise = 100 * (noise4 + noise3)  # Summing all of our noises

# Magnitude (PH) of setpoint
setpoint_magnitude = 7
# Setpoint Error Margin
setpoint_margin = 1

# Straight Line setpoint
setpoint_ref = setpoint_magnitude * np.ones(T.shape)

# Sinewave input setpoint
f = 1  # Frequency of sine wave in Herts
# setpoint_sine = setpoint_magnitude + (setpoint_magnitude + np.sin(0.05*T))

# Create a step response for the setpoint
setpoint_step = []
for i in T:
    if i < len(T) / 10:
        setpoint_step.append(0)
    else:

        setpoint_step.append(setpoint_magnitude)

# Choose Setpoint Type
setpoint_array = setpoint_ref

# # If we are using setpoint step -> Set Initial PH = 0.
# if setpoint:
#     x0 = 0
#     x1 = 0

leachate_flow = 1 * 1.157407 * np.ones(T.shape)  # leachate flow in L/S
zeros = np.zeros(T.shape)  # acid flow system input (additional to controller acid flow)
pond_level = 1570000 * np.ones(T.shape)  # Pond Level Set as Input incase we need to add plant dynamics.
# ----------------              SIMULATION CONFIG FINISH               ------------


# ----------------              SIMULATION                ------------
def ph_calc(t, x, u, params={}):
    # State Variables
    C_H = x[1]
    C_OH = x[0]
    b = C_OH - C_H

    if b > 0:
        H_p = (b / 2) * (np.sqrt((1 + (4.0 * (10 ** -14.0)) / np.power(b, 2))) - 1)
        # H_p = (1 / 2) * (-b + np.sqrt(np.power(b, 2) + 4.0 * (10 ** -14.0)))
        PH = -1 * np.log10(H_p)
    elif b < 0:
        H_p = (-b / 2) * (np.sqrt((1 + (4.0 * (10 ** -14.0)) / np.power(b, 2))) + 1)
        # H_p = (1 / 2) * (-b + np.sqrt(np.power(b, 2) + 4.0 * (10 ** -14.0)))
        PH = -1 * np.log10(H_p)
    else:
        PH = 7

    PH = PH
    u1 = u[0]

    if PRINT_INFO:
        print("PH: " + PH.__str__() + " C_H : " + C_H.__str__() + " C_OH : " + C_OH.__str__())


    return [C_OH, C_H, PH, u1]


def dosing_plant(t, x, u, params={}):
    # Dosing Equation Parameters
    C_acid_H = params.get('C_acid_H', 18) # np.float_power(10, -0.3) # concentration in moles per litre
    C_acid_OH = params.get('C_acid_OH', np.float_power(10, -13.7))
    C_leachate_H = params.get('C_leachate_H', np.float_power(10, -12.2))  # mol/L
    C_leachate_OH = params.get('C_leachate_OH', np.float_power(10, -1.8))

    # Map the states to local variables
    x1 = x[0]  # OH- Conc
    x2 = x[1]  # H+ Conc
    x3 = x[2]  # V Pond Volume

    T = params.get('TIME_PERIOD', 1)

    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    u4 = u[3]
    # volume = x[2]

    acid_flow = u[0]  # Maximum Flow Rate of 7.5 L/S
    leachate_flow = u[1]

    # dx1 = (1 / u4) * (
    #         (leachate_flow * C_leachate_OH) - ((leachate_flow + acid_flow) * x1) + (acid_flow * C_acid_OH))
    # dx2 = (1 / u4) * (
    #         (leachate_flow * C_leachate_H) - ((leachate_flow + acid_flow) * x2) + (acid_flow *
    dx1 = (1 / x3) * (
            (leachate_flow * C_leachate_OH) + (acid_flow * C_acid_OH))
    dx2 = (1 / x3) * (
            (leachate_flow * C_leachate_H) + (acid_flow * C_acid_H))

    dx3 = (leachate_flow + acid_flow)
    #
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


def calc_gain_scheduler_scale(ph):
    log_scale = np.power(10, -ph)
    [kp_scale, ki_scale] = [1 / log_scale, 1 / log_scale]
    return kp_scale


# ------------------                ADAPTIVE CONTROLLER MAIN FUNCTION                ------------------
def cstm_dosing_ctrl(t, x, u, params={}):
    # Calculate the current error
    e = -u[0]

    T = params.get('TIME_PERIOD', 1)  # Time period
    Kp = params.get('KP', 1)
    Ki = params.get('KI', 1)

    # Map the states
    xI = x[0]  # Previous Integral Accumulation error
    xPid = x[1]  # Previous PID Value

    # If adaptive Control is to be used
    if ADAPTIVE_CONTROL:
        Adaptive_Scale = calc_gain_scheduler_scale(u[1])
    else:
        Adaptive_Scale = 1

    pid_value = (Adaptive_Scale * Kp * e) + (Adaptive_Scale * Ki * xI)

    # Calculate Integral error accumulation scaled with adaptive controller scale
    if pid_value < setpoint_magnitude: # Avoid Integral Windup
        d_xI = (T * e)
    else:
        d_xI = 0

    # To-Do Add setpoint Margin
    # if e <= setpoint_margin/2

    d_xPid = pid_value - xPid

    # print(d_xPid)

    # Integral state accumulation dx_I/dt
    return [d_xI, d_xPid]


# ------------------                ADAPTIVE CONTROLLER OUTPUT FUNCTION                ------------------
def cstm_ctrl_outfcn(t, x, u, params={}):
    # Map the states
    xI = x[0]
    xPid = x[1]

    # This is the one baby!
    y = np.clip(xPid, 0, 21 / 3600)

    if PRINT_INFO:
        print("Dosing Rate: " + y.__str__())
    return y


# Transfer Function for gain scheduling
cust_ctrl = ct.NonlinearIOSystem(cstm_dosing_ctrl, outfcn=cstm_ctrl_outfcn, name='cust_ctrl', inputs=('u', 'ph'),
                                 outputs=['y0'], states=('xI', 'xPid'))

# Create our inter connected IO system for our custom control function
cstm_ctrld_dosing = ct.InterconnectedSystem((cust_ctrl, io_dosing), name='cstm_ctrld_dosing', connections=(
    ['cust_ctrl.u', '-io_dosing.y'], ['io_dosing.acid_flow', 'cust_ctrl.y0'], ['cust_ctrl.ph', 'io_dosing.y']),
                                            inplist=('cust_ctrl.u', 'cust_ctrl.ph', 'io_dosing.leachate_flow',
                                                     'io_dosing.noise', 'io_dosing.pond_volume'),
                                            inputs=('vref', 'ph', 'leachate_flow', 'noise', 'pond_volume'),
                                            outlist=['io_dosing.x1', 'io_dosing.x2', 'io_dosing.y',
                                                     'io_dosing.acid_flow'], outputs=['x1', 'x2', 'y', 'u1'])

response = ct.input_output_response(cstm_ctrld_dosing, T, [setpoint_array, zeros, leachate_flow, plant_noise,
                                                           pond_level], [x0, x1, x2, x3, x4], return_x=True,
                                    params={'KP': KP, 'KI': KI, 'T': TIME_PERIOD})

print(response)

time = response[0]
outputs = response[1]

acid_used = np.sum(TIME_PERIOD * outputs[3])
mean_error = (np.sum(outputs[2] - setpoint_array)/len(outputs[2]))

# Plot Results
plt.rcParamsDefault['lines.marker'] = '^'
plt.rcParamsDefault['lines.markersize'] = 10
# plt.rcParams['lines.linewidth'] = '3'
# plt.rcParams['lines.linestyle'] = '-'

fig1, axs = plt.subplots(2, 2)

axs[0, 0].plot(time, outputs[0], label='C_OH-')
axs[0, 0].plot(time, outputs[1], label='C_H+')
axs[0, 0].set_title("Pond Concentrations")
axs[0, 0].legend()

axs[0, 1].step(time, outputs[3], label='Acid Flow L/h')
axs[0, 1].step(time, leachate_flow, label='Leachate Flow L/h')
axs[0, 1].set_title("Acid Flows")
axs[0, 1].legend()

axs[1, 0].plot(time, outputs[2], label='PH')
axs[1, 0].plot(time, outputs[2] - setpoint_array, label='PH Error')
axs[1, 0].plot(time, setpoint_array, label='Set Point', linestyle='-')

axs[1, 0].set_title("Pond PH")
axs[1, 0].legend()

axs[1, 1].set_title("Flows and PHs =")
axs[1, 1].step(time, outputs[3], label='Acid Flow L/h')
axs[1, 1].step(time, leachate_flow, label='Leachate Flow L/h')
axs[1, 1].plot(time, setpoint_array, label='Set Point', linestyle='-')
axs[1, 1].plot(time, outputs[2], label='PH')
axs[1, 1].legend()

print("KP: KI: Adaptive: T_Period: Total Time: Acid Used: Mean Error")
print(KP.__str__() + " : " + KI.__str__() + " : " + TIME_PERIOD.__str__() + " : " +
      ADAPTIVE_CONTROL.__str__() + " : " + TIME_LENGTH.__str__() + " : "
      + acid_used.__str__() + " : " + mean_error.__str__())

plt.show()