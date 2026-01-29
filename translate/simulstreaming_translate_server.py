# TODO: duplicate with ../whisper_streaming/whisper_server.py
import line_packet
import socket

class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 32000*5*60 # 5 minutes # was: 65536

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self):
        try:
            r = self.conn.recv(self.PACKET_SIZE)
            return r
        except ConnectionResetError:
            return None



import logging
logger = logging.getLogger(__name__)


from simulstreaming_translate import *

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class TextServerProcessor:

    def __init__(self, connection, simul):
        self.connection = connection
        self.simul = simul

        self.beg = 1

        self.buffer = ""
        self.endswith_eos = False

    def receive_input_chunk(self):
        out = []
        lines = self.connection.receive_lines()
        print(lines,flush=True,file=sys.stderr)
        if lines is None:
            return None
        if self.buffer:
            lines[0] = self.buffer + lines[0]
            self.buffer = ""
        for l in lines:
            if l == "": continue
            print(l,flush=True,file=sys.stderr)
            try:
                out.append(json.loads(l))
            except (IndexError, ValueError) as e:
                self.buffer = l
        print("OUT",out,flush=True,file=sys.stderr)
        return out

#    def format_output_transcript(self,o, is_final=False):
#        # output format in stdout is like:
#        # 0 1720 Takhle to je
#        # - the first two words are:
#        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
#        # - the next words: segment transcript
#
#        # This function differs from whisper_online.output_transcript in the following:
#        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
#        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
#        # Usually it differs negligibly, by appx 20 ms.
#
#        out = []
#        print("OUTPUT:",o,file=sys.stderr)
#        status, confirmed, unconfirmed = o
#        if o:
#            beg, end = self.beg, self.beg + 1
#            if confirmed:
#                t = confirmed
#                m = "%1.0f %1.0f %s" % (beg,end,t)
#                out.append(m)
#
#                beg, end = self.beg+2, self.beg+3
#                if status == "COMPLETE":
#                    self.beg = self.beg + 2
#
#            if unconfirmed:
#                t = unconfirmed[:50] ## trim only the first 50 characters. The rest could be hallucination.
#                m = "%1.0f %1.0f %s" % (beg,end,t)
#                out.append(m)
#
#            if out and any(out[-1].endswith(x) for x in [".", ". ", "?", "? ", "!", "! "]):
#                self.endswith_eos = True
#            else:
#                if is_final:
#                    self.endswith_eos = True
#                    logger.debug("Adding '. ' to the final segment")
#                    out[-1] += ". "
#                else:
#                    self.endswith_eos = False
#            return "\n".join(out)
#        else:
#            logger.debug("No text in this segment")
#            if is_final and not self.endswith_eos:
#                logger.debug("Adding '. ' to the final segment")
#                return f"{self.beg} {self.beg+1} . "
#            return None

#    def send_result(self, o, is_final=False):
#        """o is a triple Status, confirmed, unconfirmed or None/"" """
#        msg = self.format_output_transcript(o, is_final=is_final)
#        if msg is not None and msg != "":
#            self.connection.send(msg)

    def process(self):

        def send_seq(out, row, timer, is_final=False):
            for r in format_outputs(out, row, timer, is_final=is_final):
                msg = json.dumps(r)
                self.connection.send(msg)

        # handle one client connection
        timer = SimulationTimer(comp_aware=True)
        self.simul.init()
        while True:
            a = self.receive_input_chunk()
            if a is None or a == []:
                break
            inserted = False
            try:
                for w in a:
                    print("TADY",w,flush=True,file=sys.stderr)
                    simulation_update(self.simul, w, timer, out_handler=send_seq)
            except BrokenPipeError:
                logger.info("broken pipe -- connection closed?")
                return

# TODO: duplicate with ../whisper_streaming/whisper_online_main.py
def set_logging(args,logger):
    logging.basicConfig(
        # this format would include module name:
        #    format='%(levelname)s\t%(name)s\t%(message)s')
            format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)

def main_server():
    import argparse
    parser = argparse.ArgumentParser()

    translate_args(parser)
    simulation_args(parser)

    # server options
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=43007)

    args = parser.parse_args()
    set_logging(args,logger)

    simul = simul_translator_factory(args)

    # server loop

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.host, args.port))
        s.listen(1)
        logger.info('Listening on'+str((args.host, args.port)))
        while True:
            conn, addr = s.accept()
            logger.info('Connected to client on {}'.format(addr))
            connection = Connection(conn)
            proc = TextServerProcessor(connection, simul) #, min_chunk)
            proc.process()
            conn.close()
            logger.info('Connection to client closed')
    logger.info('Connection closed, terminating.')

if __name__ == "__main__":
    main_server()