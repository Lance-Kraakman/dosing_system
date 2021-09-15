# Dosing system
import numpy as np
import control as ct
import matplotlib.pyplot as plt
import math


def ph_output(c_OH, c_H):
    b = c_OH - c_H
    C_star = (1/2)*(-b - np.sqrt(np.power(b, 2) + 40.0*(10**-14.0)))

    # Make sure log has no zeros so it doesn't complain about log(0)
    iterator = 0
    for i in C_star:
        if i == 0:
            C_star[iterator] = 0.01
        elif i < 0:
            PH = 1 * np.log10(abs(C_star))
        iterator = iterator + 1

    PH = 1*np.log10(abs(C_star))
    return PH


def dosing_plant(t, x, u, params={}):
    # Dosing Equation Parameters

    C_acid_H = params.get('C_acid_H', 0.99)
    C_acid_OH = params.get('C_acid_OH', 0.001)
    pond_volume = params.get('p_volume', 1000)

    C_leachate_H = params.get('C_leachate_H', 0.1)
    C_leachate_OH = params.get('C_leachate_OH', 0.8)

    # Map the states to local variables
    x1 = x[0]  # OH- Conc
    x2 = x[1]  # H+ Conc

    acid_flow = u[0]
    leachate_flow = u[1]

    # Compute the control action -> We are only allowed to add acid so acid must be positive
    # u[0] is acid and u[1] is our system noise

    dx1 = (1/pond_volume)*((leachate_flow*C_leachate_OH)-((leachate_flow+acid_flow)*x1)+(acid_flow*C_acid_OH))
    dx2 = (1/pond_volume)*((leachate_flow*C_leachate_H)-((leachate_flow+acid_flow)*x2)+(acid_flow*C_acid_H))

    # Return
    return [dx1, dx2]


io_dosing = ct.NonlinearIOSystem(dosing_plant, inputs=('acid_flow', 'leachate_flow'), outputs=('x1', 'x2'),
                                 states=('x1', 'x2'), name='dosing')

# Creating our data for an open loop simulation
T = np.linspace(-5, 5, 500)

u0 = 5*np.ones(T.shape)
u1 = 1*abs(np.sin(T))

x0 = 7
x1 = 7

print("Inputs:\n" + "Acid Flow Rate : Leachate Flow Rate \n" + [u0, u1].__str__())

t_vect, y_vect = ct.input_output_response(io_dosing, T, [u0, u1], [x0, x1])
ph_vect = ph_output(y_vect[0], y_vect[1])
print(t_vect, y_vect[0]-y_vect[1])
print(ph_vect)

# Plot the response
plt.figure(2)
plt.plot(t_vect, ph_vect)
plt.plot(t_vect, y_vect[0])
plt.plot(t_vect, y_vect[1])
plt.show()

# Now for the closed loop testing

