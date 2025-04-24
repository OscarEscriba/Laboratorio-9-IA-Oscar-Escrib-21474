class HMM:
    def __init__(self, states, observations, initial_prob, transition_prob, emission_prob):
        """
        Inicializa los parámetros del Modelo Oculto de Markov.

        Args:
            states (list): Lista de estados ocultos.
            observations (list): Lista de posibles observaciones.
            initial_prob (dict): Probabilidad inicial de cada estado.
            transition_prob (dict): Probabilidad de transición entre estados.
            emission_prob (dict): Probabilidad de emitir una observación dado un estado.
        """
        self.states = states
        self.observations = observations
        self.initial_prob = initial_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob

    def generate_sequence(self, length):
        """
        Genera una secuencia de observaciones basada en el HMM.

        Args:
            length (int): Longitud de la secuencia a generar.

        Returns:
            list: Una secuencia de observaciones.
        """
        import random
        current_state = random.choices(self.states, weights=list(self.initial_prob.values()), k=1)[0]
        observations_sequence = []
        for _ in range(length):
            emission_probs = self.emission_prob[current_state]
            observation = random.choices(self.observations, weights=list(emission_probs.values()), k=1)[0]
            observations_sequence.append(observation)
            transition_probs = self.transition_prob[current_state]
            current_state = random.choices(self.states, weights=list(transition_probs.values()), k=1)[0]
        return observations_sequence

    def forward(self, observations):
        """
        Implementa el paso hacia adelante del algoritmo Forward-Backward.

        Args:
            observations (list): La secuencia de observaciones.

        Returns:
            list: Una lista de diccionarios, donde cada diccionario contiene las probabilidades forward
                  para cada estado en ese paso de tiempo.
        """
        forward_probs = []
        T = len(observations)
        N = len(self.states)

        # Inicialización del primer paso forward
        f_prob = {}
        for state in self.states:
            f_prob[state] = self.initial_prob[state] * self.emission_prob[state][observations[0]]
        forward_probs.append(f_prob)

        # Recursión para los pasos forward restantes
        for t in range(1, T):
            f_prob = {}
            for current_state in self.states:
                prob = 0
                for prev_state in self.states:
                    prob += forward_probs[t - 1][prev_state] * self.transition_prob[prev_state][current_state]
                f_prob[current_state] = prob * self.emission_prob[current_state][observations[t]]
            forward_probs.append(f_prob)

        return forward_probs

    def backward(self, observations):
        """
        Implementa el paso hacia atrás del algoritmo Forward-Backward.

        Args:
            observations (list): La secuencia de observaciones.

        Returns:
            list: Una lista de diccionarios, donde cada diccionario contiene las probabilidades backward
                  para cada estado en ese paso de tiempo.
        """
        backward_probs = []
        T = len(observations)
        N = len(self.states)

        # Inicialización del último paso backward
        b_prob = {state: 1 for state in self.states}
        backward_probs.insert(0, b_prob)

        # Recursión para los pasos backward restantes
        for t in range(T - 2, -1, -1):
            b_prob = {}
            for prev_state in self.states:
                prob = 0
                for next_state in self.states:
                    prob += backward_probs[0][next_state] * self.transition_prob[prev_state][next_state] * self.emission_prob[next_state][observations[t + 1]]
                b_prob[prev_state] = prob
            backward_probs.insert(0, b_prob)

        return backward_probs

    def compute_state_probabilities(self, observations):
        """
        Calcula la probabilidad de estar en cada estado en cada paso de tiempo dada la secuencia observada.

        Args:
            observations (list): La secuencia de observaciones.

        Returns:
            list: Una lista de diccionarios, donde cada diccionario contiene la probabilidad posterior
                  de cada estado en ese paso de tiempo.
        """
        forward_probs = self.forward(observations)
        backward_probs = self.backward(observations)
        state_probs = []
        T = len(observations)

        for t in range(T):
            posterior_prob = {}
            total_prob = 0
            for state in self.states:
                posterior_prob[state] = forward_probs[t][state] * backward_probs[t][state]
                total_prob += posterior_prob[state]

            # Normalizar las probabilidades
            for state in self.states:
                posterior_prob[state] /= total_prob
            state_probs.append(posterior_prob)

        return state_probs

# 1. Definir los parámetros del modelo oculto de Markov (HMM)
states = ["Sunny", "Rainy"]
observations = ["Sunny", "Rainy"]
initial_prob = {"Sunny": 0.5, "Rainy": 0.5}
transition_prob = {
    "Sunny": {"Sunny": 0.8, "Rainy": 0.2},
    "Rainy": {"Sunny": 0.3, "Rainy": 0.7}
}
emission_prob = {
    "Sunny": {"Sunny": 0.6, "Rainy": 0.4},
    "Rainy": {"Sunny": 0.2, "Rainy": 0.8}
}

# 2. Crear una instancia de la clase HMM
hmm_model = HMM(states, observations, initial_prob, transition_prob, emission_prob)

# 3. Generar una secuencia de observaciones (opcional para pruebas)
observed_sequence = hmm_model.generate_sequence(5)
print(f"Secuencia de observaciones generada: {observed_sequence}")


# 7. Imprimir o analizar las probabilidades calculadas (ya se imprimieron)