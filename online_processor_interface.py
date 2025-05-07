class OnlineProcessorInterface:

    SAMPLING_RATE = 16000

    def insert_audio_chunk(self, audio):
        raise NotImplementedError("must be implemented in child class")
    
    def process_iter(self):
        raise NotImplementedError("must be implemented in child class")
    
    def finish(self):
        raise NotImplementedError("must be implemented in child class")