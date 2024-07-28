import numpy as np
import qiskit as q
import qiskit.circuit as circuit
import warnings
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, BravyiKitaevMapper, BravyiKitaevSuperFastMapper
from qiskit.circuit.library import TwoLocal
from qiskit_aer.backends import AerSimulator
from scipy.optimize import minimize, Bounds

class QuantumOptimizer:
    def __init__(self, rho: np.array, m: int, ancilla: int = 10, method=None, optimizer: str=None, a : float = None) -> None:
        rho = np.squeeze(rho)
        assert rho.shape[0] == rho.shape[1], "Correlation matrix is not a square matrix"
        self.n = rho.shape[0] ** 2 + rho.shape[0]
        assert m < self.n, "Cannot select more than total number of qubits"
        self.rho = rho
        self.m = m
        a = -rho.shape[0] if a is None else a
        self.a = a
        self.ancilla = ancilla
        self.transform_methods = {
            'jordan_wigner': JordanWignerMapper.map, 
            'parity': ParityMapper.map,
            'bravyi_kitaev': BravyiKitaevMapper.map, 
            'super_fast': BravyiKitaevSuperFastMapper.map
        }
        self.optimizer = optimizer
        self.simulator = AerSimulator()

        if callable(method):
            self.h = method(rho, a, m)
        else:
            self.h = self._construct_hamiltonian(rho, a, m)
        self.circuit = self._construct_circuit()
        
    # Construct exp(iHt) operator
    def _construct_hamiltonian(self, rho, a, m):
        h = q.QuantumCircuit(self.n)
        n = rho.shape[0]
        t = circuit.Parameter('time')
        _rho = np.reshape(rho, [-1])

        # Phase shifts due to Hamiltonian
        # for i in range(n * n):
        #     h.append(circuit.library.PhaseGate(_rho[i] * t), [i]) # 1st term
        #     h.append(circuit.library.PhaseGate((-2 * a) * t), [i]) # 3rd term
        #     h.append(circuit.library.PhaseGate((a) * t), [i]) # 5th term
        # for i in range(n):
        #     h.append(circuit.library.PhaseGate((-2 * a * m) * t), [n * n + i]) # 2nd term
        #     h.append(circuit.library.PhaseGate(a * t), [n * i + i]) # 4th term
        #     h.append(circuit.library.CPhaseGate((-2 * a) * t), [n * i + i, n * n + i]) # 4th term
        #     h.append(circuit.library.PhaseGate(a * t), [n * n + i]) # 4th term

        #     h.append(circuit.library.PhaseGate( a * t), [n * n + i]) # 2nd term
        #     for j in range(i + 1, n):
        #         h.append(circuit.library.CPhaseGate(2 * a * t), [n * n + i, n * n + j]) # 2nd term
        # for k in range(n):
        #     for i in range(n):
        #         h.append(circuit.library.CPhaseGate(-a * t), [i * n + k, n * n + k]) # 5th
        #         h.append(circuit.library.PhaseGate(a * t), [k * n + i]) # 3rd term
        #         for j in range(i + 1, n):
        #             h.append(circuit.library.CPhaseGate(2 * a * t), [k * n + i, k * n + j]) # 3rd term

        # Global phase
        h.append(circuit.library.GlobalPhaseGate(n * a * t))
        h.append(circuit.library.GlobalPhaseGate(m ** 2 * a * t))
        # Optimized gates
        for i in range(n):
            h.append(circuit.library.PhaseGate((-2 * a * m + 2 * a) * t), [n * n + i])
            h.append(circuit.library.PhaseGate(a * t), [n * i + i])
            h.append(circuit.library.CPhaseGate((-2 * a) * t), [n * i + i, n * n + i]) 
            for j in range(n):
                h.append(circuit.library.PhaseGate(_rho[i] * t), [i * n + j])
                h.append(circuit.library.CPhaseGate(-a * t), [j * n + i, n * n + i])
                for k in range(j + 1, n):
                    h.append(circuit.library.CPhaseGate(2 * a * t), [i * n + j, i * n + k])
            for j in range(i + 1, n):
                h.append(circuit.library.CPhaseGate(2 * a * t), [n * n + i, n * n + j])
        return h
                
    # Construct Quantum circuit
    def _construct_circuit(self):
        # ry = TwoLocal(self.n, "ry", "cz", reps=5, entanglement="full", parameter_prefix='two')
        qc = q.QuantumCircuit(self.n + self.ancilla)
        qc.h(range(self.ancilla, self.n + self.ancilla))
        # qc = qc.compose(ry, range(10, self.n + 10))
        qc = qc.compose(circuit.library.PhaseEstimation(self.ancilla, self.h)) # TODO: Ignore the middle significant digits
        # qc = qc.compose(circuit.library.PhaseEstimation(self.ancilla, test))
        qc.measure_all()
        qc.decompose().draw('mpl')
        return qc

    # Bind parameters and run circuits
    def _bind_and_run(self, circuit:q.QuantumCircuit, params:dict, shots:int):
        
        circuit = circuit.assign_parameters(params)
        comp = q.transpile(circuit, self.simulator)
        result = self.simulator.run(comp, shots=shots).result()
        return result.get_counts()

    # Objective function (not really any more since we are not using conventional optimization)
    def obj(self, params:list, shots:int):
        h = 0
        dt = 1e-2
        # for t in np.linspace(0, 10., 5):
        input = {f'two[{i}]': params[i] for i in range(len(params))}
        input.update({'time': dt})
        counts = self._bind_and_run(self.circuit, input, shots=shots)
        tally = {}
        offset = (self.rho.shape[0] * self.a + self.a * self.m ** 2) * dt / (2 * np.pi)
        for k, v in counts.items():
            sig = 1
            phase = 0
            for i in range(self.ancilla):
                sig /= 2
                if k[-i-1] == '1': phase += sig
            
            # phase += offset
            phase = phase if phase < 0.1 else phase - 1.
            # if k[0:self.n] == '011001010001': print(phase, k[self.n:], v)
            if k[0:self.n] not in tally:
                tally.update({k[0:self.n]: {phase: v}})
            else:
                tally[k[0:self.n]].update({phase: v})

            # if k[0:self.n] in tally:
            #     tally[k[0:self.n]][0] += v * phase
            #     tally[k[0:self.n]][1] += v
            # else: tally.update({k[0:self.n]: [v * phase, v]})

        return tally
    
    def optimize(self, shots:int = 131072):
        tally =  self.obj([], shots=shots)
        result = {}
        # Only consider the top 3 results
        for k, v in tally.items():
            vmax = 0
            total = 0
            num = 0
            for k2, v2 in sorted(v.items(), key= lambda item: item[1], reverse= True):
                # if vmax == 0: 
                #     vmax = v2
                # if v2 > vmax / 5: 
                if vmax < 3:
                    vmax += 1
                    total += k2 * v2
                    num += v2
            result.update({k: total / num})
            # if v[1] != 0:
                # result.update({k: v[0] / v[1]})
        # maximum = max(result, key=result.get)
        # return (maximum, result[maximum])
        return sorted(result.items(), key=lambda item: item[1], reverse=True)
    
        # opt = minimize(self.obj, np.random.uniform(size=(self.circuit.num_parameters - 1, )), bounds=Bounds(0, 1), options={'maxiter': 10000})#, method='COBYLA')
        # qc = q.QuantumCircuit(self.n)
        # qc.h(range(self.n))
        # qc = qc.compose(TwoLocal(self.n, "ry", "cz", reps=5, entanglement="full", parameter_prefix='two'))
        # qc.measure_all()
        # counts = self._bind_and_run(qc, opt.x)

        # return max(counts, key=counts.get)
        # cobyla = COBYLA()
        # cobyla.set_options(maxiter=250)
        
        # svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=cobyla)
        # svqe = MinimumEigenOptimizer(svqe_mes)
        # result = svqe.solve(self.qp)
        # return self.decode_result(result)