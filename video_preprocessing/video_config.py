from collections import defaultdict

video_config_dict = {
    "test_video": {"head_up": False, "rot90": 0, "shift": (-20, 0)},
    "DMSO_1": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    "DMSO_2": {"head_up": False, "rot90": 0, "shift": (-20, 0)},
    ###################################
    "DMSO_Dex3_1": {
        "head_up": True,
        "rot90": 1,  # not starting with fast swim?
        "shift": (-20, 0),
    },
    "DMSO_Dex3_2": {"head_up": False, "rot90": 0, "shift": (-20, 0)},  # Trop court
    "DMSO_Dex3_3": {
        "head_up": True,
        "rot90": 3,  # Problem orientation
        "shift": (-10, 0),
    },
    "DMSO_Dex3_4": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    "DMSO_Dex3_5": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    ###################################
    "DMSO_Dex5_1": {"head_up": False, "rot90": 0, "shift": (-20, 0)},
    "DMSO_Dex5_2": {
        "head_up": True,
        "rot90": 1,  # Problem orientation
        "shift": (-20, 0),
    },
    "DMSO_Dex5_3": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    "DMSO_Dex5_4": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    "DMSO_Dex5_5": {
        "head_up": True,
        "rot90": 1,  # Problem orientation
        "shift": (-20, 0),
    },
    ###################################
    "DMSO_Dex9_1": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    "DMSO_Dex9_2": {"head_up": False, "rot90": 0, "shift": (-20, 0)},
    "DMSO_Dex9_3": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    "DMSO_Dex9_4": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    "DMSO_Dex9_5": {
        "head_up": True,
        "rot90": 1,  # Problem orientation
        "shift": (-20, 0),
    },
    ###################################
    "DZP_50_µM_2": {"head_up": False, "rot90": 0, "shift": (-20, 0)},
    "DZP_50_µM_3": {"head_up": True, "rot90": 0, "shift": (-10, 0)},
    "DZP_50_µM_4": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    ###################################
    "DZP_50_µM_Dex3_2": {
        "head_up": True,
        "rot90": 3,  # Problem orientation
        "shift": (-20, 0),
    },
    "DZP_50_µM_Dex3_3": {"head_up": False, "rot90": 0, "shift": (-20, 0)},
    "DZP_50_µM_Dex3_4": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    "DZP_50_µM_Dex3_5": {
        "head_up": True,
        "rot90": 3,  # late reaction
        "shift": (-20, 0),
    },
    ###################################
    "DZP_50_µM_Dex5_1": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    "DZP_50_µM_Dex5_2": {"head_up": False, "rot90": 0, "shift": (-20, 0)},
    "DZP_50_µM_Dex5_3": {"head_up": False, "rot90": 0, "shift": (-20, 0)},
    "DZP_50_µM_Dex5_4": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    "DZP_50_µM_Dex5_5": {
        "head_up": True,  # Problem orientation
        "rot90": 1,
        "shift": (-20, 0),
    },
    ###################################
    "DZP_50_µM_Dex9_1": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    "DZP_50_µM_Dex9_2": {"head_up": True, "rot90": 0, "shift": (-20, 0)},  # Trop court
    "DZP_50_µM_Dex9_3": {"head_up": True, "rot90": 0, "shift": (-20, 0)},  # Trop court
    "DZP_50_µM_Dex9_4": {"head_up": True, "rot90": 0, "shift": (-20, 0)},
    ###################################
}
