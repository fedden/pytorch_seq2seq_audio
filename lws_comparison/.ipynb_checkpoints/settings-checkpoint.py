from utils import str_to_bool

class Settings():
    
    def __init__(self, args):
        
        # Used for future se ssions and file names.
        self.epoch = 0
        self.loss = 0
        
        # What is the amount of fft bins?
        self.fft_size = int(args['fft_size'])
        
        # Do we use lws for mags (or librosa) ?
        self.lws_mags = str_to_bool(args['lws_mags'])
        
        # Which phase estimation method do we use?
        if args['phase_estimation_method'] == 'lws':
            self.griffin_lim_phase = False
            self.lws_phase = True
            self.vocoder_phase = False
            
        elif args['phase_estimation_method'] == 'griffin_lim':
            self.griffin_lim_phase = True
            self.lws_phase = False
            self.vocoder_phase = False
        
        elif args['phase_estimation_method'] == 'vocoder':
            self.griffin_lim_phase = False
            self.lws_phase = False
            self.vocoder_phase = True
            
        # How many iterations of griffin lim phase estimation?
        if args['griffin_lim_iterations'] != None:
            self.griffin_lim_iterations = int(args['griffin_lim_iterations'])  
        else:
            self.griffin_lim_iterations = None
            
        # Do we want lws 'perfect reconstruction' padding?
        self.perfect_reconstruction = str_to_bool(args['perfect_reconstruction'])
        
        # Do we want to use lws music or speech mode?
        self.mode = args['lws_mode']
        
        # Which cuda device to run the model on?
        if args['cuda_device'] == None:
            self.cuda_device = 0  
        else:
            self.cuda_device = int(args['cuda_device'])
            
        # What is the size of the input and output of the model?
        self.sequence_length = 150
        
        # How many examples are fed to the model?
        self.batch_size = 8
        
        # See the audio dataset class.
        if args['overlap_ratio'] == None:
            self.overlap_ratio = 0.0
        else:
            self.overlap_ratio = float(args['overlap_ratio'])
            
        # A path to a valid empty folder to save state in.
        self.save_path = str(args['save_path']) if args['save_path'] else None
        
        # A path to a valid non-empty folder to load things from.
        self.load_path = str(args['load_path']) if args['load_path'] else None
        
        # How many epochs to train for? Enter zero for just inference.
        self.number_epochs = int(args['epochs'])
        
        # All derived from FFT size.
        self.feature_size = (self.fft_size // 2) + 1
        self.input_size = self.hidden_size = self.feature_size
        self.hop_length = self.fft_size // 4
        
        # The audio file we train on.
        self.path = args['audio_path']
        
        # Set this to check that reconstructions are working sensibly.
        self.sanity_check = args['sanity_check']
        
        # How many layers deep will the encoder be?
        if args['encoder_layers'] == None:
            self.encoder_layers = 3
        else:
            self.encoder_layers = int(args['encoder_layers'])
        
        # How many layers deep will the decoder be?
        if args['decoder_layers'] == None:
            self.decoder_layers = 3
        else:
            self.decoder_layers = int(args['decoder_layers'])
            
        # Does the RNN's have dropout?
        if args['dropout'] == None:
            self.dropout = 0.0
        else:
            self.dropout = float(args['dropout'])
            
        # What is the baseline learning rate? (the decoder has five times this.)
        if args['learning_rate'] == None:
            self.learning_rate = 0.00001
        else:
            self.learning_rate = float(args['learning_rate'])
        
        # How many frames do we run inference for?
        if args['inference_length'] == None:
            self.inference_length = 20
        else:
            self.inference_length = int(args['inference_length'])
        
        # Sanity checks.
        valid_phase_options = ['lws', 'griffin_lim', 'vocoder']
        correct_phase = args['phase_estimation_method'] in valid_phase_options
        if not correct_phase:
            err_str = 'Please ensure phase_estimation_method is passed '
            err_str += 'with lws, griffin_lim or vocoder'
            raise ValueError(err_str)
        
        incomplete_lws_params = self.mode is None or self.perfect_reconstruction is None
        incomplete_gl_params = self.griffin_lim_iterations is None
        if not self.griffin_lim_phase and incomplete_lws_params:
            lws_str =  'Mode and perfect reconstruction parameters '
            lws_str += 'must both be set.'
            raise ValueError(gl_str)

        if self.griffin_lim_phase and incomplete_gl_params:
            gl_str =  'you must set a value for the griffin lim '
            gl_str += 'iterations. 100 is a good value.'
            raise ValueError(gl_str)
        