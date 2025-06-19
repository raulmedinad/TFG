# Código basado en https://github.com/dwave-examples/coordinated-multipoint-notebook
# Raúl Medina del Campo
# UAM junio 2025
import numpy as np
import dimod
import json

from helpers.filters import ALL_METHODS, apply_filters, compare_signals, create_filters
from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSampler, TabuSampler
from dwave.system import FixedEmbeddingComposite
from helpers.network import configure_network, create_channels, print_network_stats, simulate_signals

from qci_client import QciClient


token = "356d1541cd56e32fda0bbdcffa7b1157"
api_url = "https://api.qci-prod.com"
qclient = QciClient(api_token=token, url=api_url)

def qci_solver(Q, runs,transmitted_symbols):
    bqm_qci = {
        'file_name': "smallest_objective.json",
        'file_config': {'qubo': {"data": Q}}  # Q debe ser serializable a JSON
    }

    # Subir el archivo con el QUBO
    response_json = qclient.upload_file(file=bqm_qci)

    # Crear el cuerpo del trabajo para QCI
    job_body = qclient.build_job_body(
        job_type="sample-qubo",
        qubo_file_id=response_json['file_id'],
        job_params={
            "device_type": "dirac-1",
            "num_samples": runs
        }
    )

    # Procesar el trabajo
    job_response = qclient.process_job(job_body=job_body)
    soluciones_qci = job_response["results"]["solutions"]

    # Convertir soluciones a formato Ising (-1, +1)
    soluciones_qci = np.array(soluciones_qci)
    ising_solution = [2 * x - 1 for x in soluciones_qci]

    # Comparar con símbolos transmitidos
    results = []
    pred = np.array(transmitted_symbols.flatten())
    labels = np.array(ising_solution)  # array 2D
    if runs == 1:
        accuracy = 100 * np.mean(pred == labels[i].flatten())


    for i in range(runs):
        accuracy = 100 * np.mean(pred == labels[i].flatten())
        results.append(accuracy)

    return results

def obtain_QUBOmatrix(bqm):
    bqm_qubo = bqm.to_qubo()
    qubo_dict = bqm_qubo[0]
    
    # Determinar el tamaño de la matriz (índice máximo + 1)
    n = max(max(i, j) for i, j in qubo_dict.keys()) + 1

    # Crear matriz NumPy de ceros
    Q = np.zeros((n, n))

    # Rellenar la matriz
    for (i, j), value in qubo_dict.items():
        if i == j:
            Q[i, i] = value  # Términos lineales (diagonal)
        else:
            Q[i, j] = 0.5 * value  # Términos cuadráticos (fuera de la diagonal)
            Q[j, i] = 0.5 * value  # Matriz simétrica

    return Q



def loop_comparisons_qci(qpu: dimod.sampler = None, 
                     runs: int = 5, 
                     network_size: int = 16, 
                     snr: float = 5, 
                     ratio: float = 1.5, 
                     solvers: list = None, qci_results: list = None) -> dict:


    if runs < 3:
        raise ValueError(f"Minimum supported runs is 3; got {runs}.")

    if solvers is None:
        solvers = ['zero_forcing', 'matched_filter', 'MMSE', "QCI"]

    network, _ = configure_network(
        network_size=network_size, 
        ratio=ratio, qpu=qpu)
    print_network_stats(network)


    sampler_sa = SimulatedAnnealingSampler()
    sampler_sd = SteepestDescentSampler()
    sampler_tabu = TabuSampler()
      
    SNR=snr

    results = {'QCI': qci_results}
    results.update({key: [] for key in solvers})

    print("Run number: ", end="")

    for run in range(runs):

        print(f"{run}, ", end="")
        channels, channel_power =  create_channels(network)
        y, transmitted_symbols = simulate_signals(channels, channel_power, SNRb=SNR)

        methods = set(ALL_METHODS).intersection(solvers)
        filters = create_filters(channels, methods=methods, snr_over_nt=SNR)

        bqm = dimod.generators.wireless.coordinated_multipoint(
            network, 
            modulation='BPSK', 
            transmitted_symbols=transmitted_symbols, 
            F_distribution=('binary','real'), 
            F=channels,
            y=y)

        def avg(ss, ts):
            return round(100 * sum(np.array(list(ss.first.sample.values())) == ts.flatten()) / len(ts))
        

        v = apply_filters(y, filters)
        filter_results = compare_signals(v, transmitted_symbols, silence_printing=True)
        for filter in methods:
            results[filter].append(filter_results[f'{filter}'])


        if 'SA' in solvers:
            sampleset_sa = sampler_sa.sample(bqm, num_reads=1, num_sweeps=150)
            results['SA'].append(round(100*sum(np.array(list(sampleset_sa.first.sample.values())) == transmitted_symbols.flatten())/len(transmitted_symbols)))
        if 'QCI' in solvers:
            sampleset_sd = sampler_sd.sample(bqm, num_reads=1)
            results['QCI'].append(qci_results[run])

        if 'SD' in solvers:
            sampleset_sd = sampler_sd.sample(bqm, num_reads=1)
            results['SD'].append(round(100*sum(np.array(list(sampleset_sd.first.sample.values())) == transmitted_symbols.flatten())/len(transmitted_symbols)))

        if 'tabu' in solvers:
            sampleset_tabu = sampler_tabu.sample(bqm, num_reads=1, timeout=30)
            results['tabu'].append(round(100*sum(np.array(list(sampleset_tabu.first.sample.values())) == transmitted_symbols.flatten())/len(transmitted_symbols)))

    print("\n")
	
    return results



