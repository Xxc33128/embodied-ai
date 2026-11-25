import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Offline tool generation only; no robot stepping or model calls.')
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
options=parser.parse_args()
sys.path.insert(0,str(options.project.resolve()))
from robosuite_bowl import RobosuiteBowlEmbodiment
from inspect_robots_agent._tools import build_toolset
emb=RobosuiteBowlEmbodiment(cameras=False)
t=build_toolset(emb.info.action_space,emb.info.observation_space,control_hz=20,max_speed_frac=.25,images='always')
base={'eef_state':np.array([-.1,0.,.95,0.,.99])}
rows=[]
for targets in [{'x':0.},{'x':0.,'gripper':1},{'x':0.,'gripper':.5},{'x':0.,'gripper':0}]:
 obs=SimpleNamespace(state=base)
 res=t.execute(SimpleNamespace(id='probe',name='move_to',arguments=json.dumps({'targets':targets,'note':'offline timing probe'})),obs)
 assert res.error is None,res.error
 rows.append({'measured_state':base['eef_state'].tolist(),'targets':targets,'chunk_steps':len(res.chunk.actions)})
print(json.dumps({'step_limits':t._step_limits.tolist(),'cases':rows},indent=2))
options.output.write_text(json.dumps({'step_limits':t._step_limits.tolist(),'cases':rows},indent=2)+'\n')
