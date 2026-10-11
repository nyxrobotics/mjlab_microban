"""Whole-body motion tracking for Microban (Teleopit's tracking task, ported).

One policy follows retargeted human motion (LAFAN1 through GMR): the actor
sees its last 10 frames (the current one last) of the reference's joints and
trunk motion and of its own state, the neck follows the reference's neck
angles, and the 18 body joints take ``HOME + action``.  The training side of
Teleopit f926386 (https://github.com/BotRunner64/Teleopit, Apache-2.0) was
adapted for mjlab 1.3 and the Microban robot; each ported file names its
source.
"""

# mjlab_microban.tasks registers this task and this package uses two of its
# modules (the HMD limits, the HOME stamp).  Loading the task package first
# lets any wholebody module be imported first without breaking the
# registration of every Microban task.
import mjlab_microban.tasks  # noqa: F401
