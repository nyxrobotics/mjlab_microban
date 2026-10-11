# GMR's registry (general_motion_retargeting/params.py), reduced to what
# motion_retarget.py reads.  Both tables are filled at run time: the robot
# model is built from robot.xml (retarget/model.py) and the IK configuration
# is made per human skeleton (retarget/core.py).
ROBOT_XML_DICT = {}
IK_CONFIG_DICT = {}
