import os
from openllmetry import init, tracer

def initialize_tracing():
    """Initialize OpenLLMetry tracing for the application."""
    # Get API key from environment variable or use placeholder
    api_key = os.getenv("TRACELOOP_API_KEY")
    
    if not api_key:
        print("Warning: TRACELOOP_API_KEY not set. Set it to enable tracing to Traceloop.")
        # Initialize without API key for local development
        init()
    else:
        # Initialize with API key for production
        init(api_key=api_key)
    
    print("OpenLLMetry tracing initialized")
    return tracer

# Initialize tracing when module is imported
tracer_instance = initialize_tracing()