EXPECTED_FEATURE_COUNT = 116

# General 

AGENT_ROLES = {

    # Duelist
    "iso": "D",
    "jett": "D",
    "neon": "D",
    "phoenix": "D",
    "raze": "D",
    "reyna": "D",
    "waylay": "D",
    "yoru": "D",
    

    # Initiator
    "breach": "I",
    "fade": "I",
    "gekko": "I",
    "kayo": "I",
    "skye": "I",
    "sova": "I",
    "tejo": "I",

    # Controller
    "astra": "C",
    "brimstone": "C",
    "clove": "C",
    "harbor": "C",
    "miks": "C",
    "omen": "C",
    "viper": "C",
    
    # Sentinel
    "chamber": "S",
    "cypher": "S",
    "deadlock": "S",
    "killjoy": "S",
    "sage": "S",
    "veto": "S",
    "vyse": "S",
}

PLAYSTYLE_MAP = {

    "1D-2I-1C-1S": "STANDARD",

    "1D-1I-2C-1S": "CONTROL",
    "1D-2I-2C-0S": "CONTROL",
    "2D-1I-2C-0S": "CONTROL",

    "2D-1I-1C-1S": "AGGRESSIVE",
    "2D-2I-1C-0S": "AGGRESSIVE",

    "1D-1I-1C-2S": "UTILITY_HEAVY",
    "0D-2I-2C-1S": "UTILITY_HEAVY",
    "0D-2I-1C-2S": "UTILITY_HEAVY",
}

# Team

AGENT_ROLE_MAP = {

    # Duelist
    "iso": "duelist",
    "jett": "duelist",
    "raze": "duelist",
    "reyna": "duelist",
    "yoru": "duelist",
    "neon": "duelist",
    "phoenix": "duelist",
    "waylay": "duelist",

    # Initiator
    "breach": "initiator",
    "fade": "initiator",
    "gekko": "initiator",
    "kayo": "initiator",
    "skye": "initiator",
    "sova": "initiator",
    "tejo": "initiator",

    # Controller
    "astra": "controller",
    "brimstone": "controller",
    "clove": "controller",
    "harbor": "controller",
    "miks": "controller",
    "omen": "controller",
    "viper": "controller",

    # Sentinel
    "chamber": "sentinel",
    "cypher": "sentinel",
    "deadlock": "sentinel",
    "killjoy": "sentinel",
    "sage": "sentinel",
    "veto": "sentinel",
    "vyse": "sentinel"
}  