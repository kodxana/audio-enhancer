INPUT_SCHEMA = {
    'input_file_url': {
        'type': str,
        'required': True
    },
    'ddim_steps': {
        'type': int,
        'required': False,
        'default': 50,
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None,  # If None, a random seed will be generated
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 3.5,
    },
}
