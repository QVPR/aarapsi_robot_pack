#!/usr/bin/env python3

import sys

def on_server_loaded(server_context):
    ''' If present, this function is called when the server first starts. '''
    print("Boot!")
    print("")
    sys.stdout.flush()
    pass

def on_server_unloaded(server_context):
    ''' If present, this function is called when the server shuts down. '''
    print("Unloaded!")
    print("")
    sys.stdout.flush()
    pass

def on_session_created(session_context):
    ''' If present, this function is called when a session is created. '''
    print("New Session!")
    print("")
    sys.stdout.flush()
    pass

def on_session_destroyed(session_context):
    ''' If present, this function is called when a session is closed. '''
    print("Destroyed!")
    print("")
    sys.stdout.flush()
    pass