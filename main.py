import numpy as np
import control as ct
import matplotlib.pyplot as plt
import math

ITERATIOINS = 100 # Max Iterations



def main_app():
    # Use a breakpoint in the code line below to debug your script.
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_app()


def dosing_sys_rhs(t, x, u, params):
    # Dosing Equation Parameters

    # Map the states to local variables
    x1 = 1
    acid = u[0]
    noise = u[1]

    # Compute the control action -> We are only allowed to add acid so acid must be positive
    # u[0] is acid and u[1] is our system noise

    dx1 = 0

    # Return
    return dx1


io_dosing = ct.NonlinearIOSystem(dosing_sys_rhs, inputs=('acid', 'noise'), outputs='x1', states='x1', name='dosing')

x0 = 1
T = np.linspace(0, ITERATIOINS-1, ITERATIOINS)
u1 = 5 * np.ones(T.shape)
u2 = 1 * np.ones(T.shape)

u2 = np.random.normal(0, 0.1, ITERATIOINS-1)
u2 = np.insert(u2, 0, 0)

t_vect, y_vect = ct.input_output_response(io_dosing, T, [u1, u2], x0)

print(u1)
print(u2)
print(y_vect)

# Plot the response
# plt.figure(1)
# plt.plot(t_vect, y_vect[0])
# plt.show()

