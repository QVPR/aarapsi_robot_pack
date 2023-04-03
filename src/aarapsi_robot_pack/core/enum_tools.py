from enum import Enum

def enum_contains(value, enumtype):
# Return true/false if the value exists within enum
    for i in enumtype:
        if i.value == value:
            return True
    return False

def enum_get(value, enumtype):
# Return enumtype corresponding to value if it exists (or return None)
    for i in enumtype:
        if i.value == value:
            return i
    return None


def enum_value_options(enumtype, skip=[None]):
# Return lists of an enumtype's values and a cleaned string variant for printing purposes
    if isinstance(skip, enumtype):
        skip = [skip]
    options = []
    options_text = []
    for i in enumtype:
        if i in skip: 
            continue
        options.append(i.value)
        options_text.append(i.name)
    return options, str(options_text).replace('\'', '')

def enum_value(enum_in):
    return enum_in.value

def enum_name(enum_in):
    return str(enum_in.name)