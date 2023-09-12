from rmatrix_bloch_se import *
import time
import plotly, plotly.subplots


def initCCsystem(
    # Potential parameters for diagonal components
    V0=60,  # real potential strength, MeV
    W0=0,  # imag potential strength, MeV
    R=4,  # Woods-Saxon potential radius and Gaussian transition peak position.
    a=0.5,  # Woods-Saxon potential diffuseness
    # transition potentials have different depths compared to diagonal terms
    # and use surface peaked Gaussian form factors rather than Woods-Saxons
    # strength of real transition potentials MeV:
    beta=20,  # DWBA terms strenght, MeV
    gamma=5,  # Beyond-DWBA terms strenght
    # strength of imaginary transition potentials MeV
    betaImaginary=0,
    gammaImaginary=0,
    # range of transition potential
    alpha=0.5,  # diffuseness, fm
    collision_energy=13,  # bombarding energy in COM frame
    level_energies=[
        0,
        2,
        4,
    ],  # energy difference from groud state in each channel [MeV]. This is called $\Delta_n$ in the introductory text.
):
    """
    3 level system example with local diagonal and transition potentials and neutral particles.
    """

    # print(V0,W0,beta,gamma)
    params_diag = (V0, W0, R, a)

    nodes_within_radius = 10

    system = ProjectileTargetSystem(
        incident_energy=collision_energy,
        reduced_mass=939,  # MeV / c^2. Reduced mass of the system.
        channel_radius=5 * (2 * np.pi),
        num_channels=len(level_energies),  # number_of_states
        level_energies=level_energies,
    )

    l = 0  # Angular momentum quantum number

    matrix = np.empty((system.num_channels, system.num_channels), dtype=object)

    for i in range(system.num_channels):
        for j in range(system.num_channels):
            if i == j:  # diagonal potentials are just Woods-Saxons
                interaction = lambda r: woods_saxon_potential(r, params_diag)
            elif i == 0 or j == 0:  # off diagonal potential terms, DWBA
                interaction = lambda r: surface_peaked_gaussian_potential(
                    r, (beta, betaImaginary, R, alpha)
                )
            else:  # off diagonal potential terms, beyond-DWBA
                interaction = lambda r: surface_peaked_gaussian_potential(
                    r, (gamma, gammaImaginary, R, alpha)
                )

            matrix[i, j] = RadialSEChannel(
                l=l,
                system=system,
                interaction=interaction,
                threshold_energy=system.level_energies[i],
            )

    # solver_lm =
    return [system, LagrangeRMatrix(40, system, matrix)]


def timeCCRmatrix(plot=False, **kwargs):
    timestart = time.time()

    (system, solver_lm) = initCCsystem(**kwargs)
    # H = solver_lm.bloch_se_matrix()

    # get R and S-matrix, and both internal and external soln
    R, S, uint = solver_lm.solve_wavefunction()

    s_values = np.linspace(0.05, system.channel_radius, 500)

    if plot:
        figure = plotly.subplots.make_subplots(
            x_title=r"$s_n = k_n r$",
            y_title=r"$u (s) $ [a.u.]",
        )
        figure.update_layout(title=str(kwargs))

        for channel in range(system.num_channels):
            channel_u_values = uint[channel].uint(units="s")(s_values)

            for yset, name in zip(
                (np.real(channel_u_values), np.imag(channel_u_values)),
                ("n=" + str(channel) + ", real", "n=" + str(channel) + ", imaginary"),
            ):
                figure.add_trace(
                    plotly.graph_objs.Scatter(x=s_values, y=yset, name=name)
                )

        figure.write_image("u.png", width=1920, height=1080)

    timestop = time.time()

    return timestop - timestart


if __name__ == "__main__":
    # print(timeCCRmatrix(plot=True, V0=60, W0=0, beta=60, gamma=60, collision_energy=50, level_energies=[0, 12, 20]))
    print(
        timeCCRmatrix(
            plot=False,
            V0=60,
            W0=20,
            beta=20,
            gamma=5,
            collision_energy=13,
            level_energies=[0, 2, 4],
        )
    )
