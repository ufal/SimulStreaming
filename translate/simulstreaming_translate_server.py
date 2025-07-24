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

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class TextServerProcessor:

    def __init__(self, connection, simul):
        self.connection = connection
        self.simul = simul

        self.last_end = None
        self.is_first = True

        self.buffer = ""

    def receive_input_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
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
            m = "0 "+l
            print(l,flush=True,file=sys.stderr)
            try:
                for _,_,_,w in yield_input_line(m):
                    out.append(w)
            except (IndexError, ValueError) as e:
                self.buffer = l
        print("OUT",out,flush=True,file=sys.stderr)
        return out

    def format_output_transcript(self,o):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if o:
            beg, end = 1,2
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print("%1.0f %1.0f %s" % (beg,end,o),flush=True,file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg,end,o)
        else:
            logger.debug("No text in this segment")
            return None

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        # handle one client connection
        self.simul.init()
        while True:
            a = self.receive_input_chunk()
            if a is None or a == []:
                break
            for w in a:
                print("TADY",w,flush=True,file=sys.stderr)
                if w.startswith(" "):
                    w = w[1:]
                    self.simul.insert_suffix(w)
                else:
                    self.simul.insert(w)
            o = self.simul.process_iter_aware()
            try:
                self.send_result(o)
            except BrokenPipeError:
                logger.info("broken pipe -- connection closed?")
                break

#        o = online.finish()  # this should be working
#        self.send_result(o)


# TODO: duplicate with ../whisper_streaming/whisper_online_main.py
def set_logging(args,logger):
    logging.basicConfig(
        # this format would include module name:
        #    format='%(levelname)s\t%(name)s\t%(message)s')
            format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)

from simulstreaming_translate import *
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