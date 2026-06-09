import sys
class ASRBase:

    sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                # "" for faster-whisper because it emits the spaces when neeeded)

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)


    def load_model(self, modelsize, cache_dir):
        raise NotImplemented("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def warmup(self, audio, init_prompt=""):
        return self.transcribe(audio, init_prompt)

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")

    def set_translate_task(self):
        raise NotImplemented("must be implemented in the child class")
    

class OnlineProcessorInterface:

    SAMPLING_RATE = 16000

    def insert_audio_chunk(self, audio):
        raise NotImplementedError("must be implemented in child class")
    
    def process_iter(self):
        raise NotImplementedError("must be implemented in child class")
    
    def finish(self):
        raise NotImplementedError("must be implemented in child class")