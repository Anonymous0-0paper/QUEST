class Connection:
    def __init__(self, bandwidth: float, min_latency: float, max_latency: float):
        self.bandwidth = bandwidth  # in Gbps
        self.min_latency = min_latency  # in ms
        self.max_latency = max_latency  # in ms
