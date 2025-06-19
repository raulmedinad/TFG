# Código basado en https://github.com/dwave-examples/coordinated-multipoint-notebook
# Raúl Medina del Campo
# UAM junio 2025

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import dwave_networkx as dnx

styles = {'QPU': 'b*', 'zero_forcing': 'g^', 'MMSE': 'mv', 'matched_filter': 'y>', 
    'SA': 'rx', 'SD': 'cp', 'tabu': 'kD', 'QCI': 'b*'}

def draw_network(network: nx.Graph):
    """Plot the given network.

    Args:
        network: Network graph.
    """
    if len(network) < 1000:
        fig = plt.figure(figsize=(8, 3))
    else:
        fig = plt.figure(figsize=(10, 10))

    tx = nx.get_node_attributes(network, 'num_transmitters')
    rx = nx.get_node_attributes(network, 'num_receivers')

    nx.draw_networkx(network, pos={n: n for n in network.nodes()}, 
        node_color = ['r' if tx[n] else 'g' if rx[n] else 'w' for n in network.nodes()], 
        with_labels=False, node_size=50)
    plt.show()

def draw_loop_comparison(results: dict, 
                         network_size: int = 16, 
                         ratio: float = 1.5, 
                         SNRb: float = 5):
    """Plot BER (Bit Error Rate) results from decoding comparisons.

    Args:
        results: Dictionary with precision percentages.

        network_size: Size of the network's underlying lattice.

        ratio: Ratio of transmitters to receivers.

        SNRb: Signal-to-noise ratio per bit.
    """
    # Calcular BER a partir de los resultados de precisión
    BER = {key: [100 - val for val in results[key]] for key in results}

    fig = plt.figure(figsize=(8, 3))

    for key in BER:
        plt.plot(BER[key], styles[key], label=key, markersize=5)
        plt.plot(len(BER[key]) * [np.mean(BER[key])], styles[key][0])

    plt.xlabel("Run")
    plt.ylabel("Bit Error Rate [%]")
    plt.legend()
    plt.xticks(range(len(next(iter(BER.values())))))  # Usar la longitud de cualquier lista en BER
    plt.suptitle(f"Network size={network_size}, Tx/Rx≈{ratio}, SNRb={SNRb}")
    plt.show()

def draw_instantiation_times(times: dict, network_sizes: dict):
    """Plot results of decoding comparisons.

    Args:
        times: Instantiations times, as a dict.

        network_sizes: Sizes of the underlying lattice.
    """

    fig = plt.figure(figsize=(8, 3))

    for key in times:
        plt.plot(times[key], styles[key] + '-', label=key, markersize=5)

    plt.xlabel("Network Size")
    plt.ylabel("Instantiation Time [ms]")
    plt.legend()
    plt.xticks(range(len(times[key])), labels=network_sizes)    
    plt.suptitle(f"Instantiation Times for Standard Linear Filters")
    plt.show()


def plot_ber_3d(results_array, ber_threshold=10.0):
    """
    Genera un gráfico 3D de BER (Bit Error Rate) en función de SNRb y ratio Tx/Rx.

    Args:
        results_array (np.ndarray): Array de forma (n, 3) donde cada fila es [SNRb, ratio, BER(%)].
        ber_threshold (float): Umbral de referencia para la BER en porcentaje.
    """
    # Extraer columnas
    SNRb = results_array[:, 0]
    ratio = results_array[:, 1]
    BER = results_array[:, 2] / 100  # Convertir a proporción

    # Crear figura y eje 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Barras verticales
    ax.bar3d(SNRb, ratio, np.zeros_like(BER), dx=0.5, dy=0.1, dz=BER, color='green', alpha=0.6)

    # Puntos en la cima de las barras
    ax.scatter(SNRb, ratio, BER, color='red', marker='x')

    # Superficie plana de referencia
    snrb_range = np.linspace(min(SNRb), max(SNRb), 10)
    ratio_range = np.linspace(min(ratio), max(ratio), 10)
    snrb_grid, ratio_grid = np.meshgrid(snrb_range, ratio_range)
    ber_plane = np.full_like(snrb_grid, ber_threshold / 100)  # Convertir umbral a proporción

    ax.plot_surface(snrb_grid, ratio_grid, ber_plane, color='salmon', alpha=0.3)

    # Etiquetas
    ax.set_xlabel("SNRb")
    ax.set_ylabel("Tx/Rx")
    ax.set_zlabel("Bit Flips in Transmission [%]")
    plt.tight_layout()
    plt.show()

