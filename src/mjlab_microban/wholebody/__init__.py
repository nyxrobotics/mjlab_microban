"""Whole-body motion tracking for Microban (Teleopit's tracking task, ported).

One policy follows retargeted human motion (LAFAN1 through GMR): the actor
sees the reference's joints and trunk motion and its own last 10 frames, the
neck follows the reference as the HMD would move it, and the 18 body joints
take ``HOME + action``.  The training side of Teleopit f926386
(https://github.com/BotRunner64/Teleopit, Apache-2.0) was adapted for mjlab
1.3 and the Microban robot; each ported file names its source.
"""
